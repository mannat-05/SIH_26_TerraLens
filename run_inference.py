#!/usr/bin/env python3
"""
Terra Lens — Pipeline B Inference Script
=========================================

Loads the trained SwinIR (fine-tuned, 4-band, edge-guided) + serial FRN
checkpoints and runs ×2 super-resolution on user-supplied images.

Supported input formats:
  • GeoTIFF (.tif/.tiff) — expects 4 bands in B2/B3/B4/B8 order
  • NumPy (.npy)         — shape [4, H, W], float32, reflectance [0,1]
  • PyTorch (.pt)        — shape [4, H, W]
  • Standard image (.png/.jpg/.jpeg/.bmp) — will be converted to
    pseudo-4-band by duplicating red channel as NIR

Usage:
  python run_inference.py --input path/to/image_or_dir [--output results/]
  python run_inference.py --input sample.tif --save-tensor

Requirements:
  pip install torch numpy matplotlib
  pip install rasterio          # only if feeding GeoTIFF files
  git clone https://github.com/JingyunLiang/SwinIR.git   # or have it locally

Author: Team TechTonic
"""

import os
import sys
import math
import glob
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. Parse command-line arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Terra Lens Pipeline B — Inference (×2 super-resolution)"
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to a single image file or a directory of images."
    )
    p.add_argument(
        "--output", "-o", default="./results",
        help="Directory to write SR outputs (default: ./results)."
    )
    p.add_argument(
        "--bundle-dir", "-b",
        default=None,
        help="Path to the pipeline B bundle directory containing the .pt "
             "checkpoint files. Auto-detected if omitted."
    )
    p.add_argument(
        "--swinir-repo", default=None,
        help="Path to a local clone of github.com/JingyunLiang/SwinIR. "
             "If omitted the script will try to clone it automatically."
    )
    p.add_argument(
        "--device", default=None,
        help="Device to run on (e.g. 'cuda', 'cpu'). Auto-detected if omitted."
    )
    p.add_argument(
        "--save-tensor", action="store_true",
        help="Also save the raw SR output as a .npy file."
    )
    p.add_argument(
        "--no-frn", action="store_true",
        help="Run SwinIR only, skip the FRN refinement head."
    )
    p.add_argument(
        "--tile-size", type=int, default=0,
        help="If > 0, process the image in tiles of this size to save GPU "
             "memory (useful for very large GeoTIFFs). 0 = process whole image."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Locate the SwinIR source & the checkpoint bundle
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

def find_bundle_dir(override=None):
    """Return the path to the pipeline B bundle directory."""
    if override and Path(override).is_dir():
        return Path(override)
    # Check common relative locations
    candidates = [
        SCRIPT_DIR / "terra_lens_pipelineB_bundle (1)",
        SCRIPT_DIR / "terra_lens_pipelineB_bundle",
        SCRIPT_DIR / "checkpoints",
    ]
    for c in candidates:
        if c.is_dir() and (c / "swinir_finetuned_best.pt").exists():
            return c
    raise FileNotFoundError(
        "Could not auto-detect the checkpoint bundle directory. "
        "Pass --bundle-dir explicitly."
    )


def ensure_swinir_repo(override=None):
    """Make sure we can `from models.network_swinir import SwinIR`."""
    if override and Path(override).is_dir():
        sys.path.insert(0, str(override))
        return str(override)
    # Check a few common locations
    candidates = [
        SCRIPT_DIR / "SwinIR",
        Path.cwd() / "SwinIR",
    ]
    for c in candidates:
        if (c / "models" / "network_swinir.py").exists():
            sys.path.insert(0, str(c))
            return str(c)
    # Clone automatically
    repo_dir = SCRIPT_DIR / "SwinIR"
    print("[INFO] Cloning official SwinIR repo …")
    ret = os.system(f'git clone -q https://github.com/JingyunLiang/SwinIR.git "{repo_dir}"')
    if ret != 0:
        raise RuntimeError(
            "Failed to clone SwinIR. Either clone it manually or pass "
            "--swinir-repo /path/to/SwinIR."
        )
    sys.path.insert(0, str(repo_dir))
    return str(repo_dir)


# ---------------------------------------------------------------------------
# 2. Model architecture components (must match training exactly)
# ---------------------------------------------------------------------------

# ---- Edge-guided modulation ----
class SobelEdge(nn.Module):
    def __init__(self, n_bands):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1))
        self.n_bands = n_bands

    def forward(self, x):
        gx = F.conv2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), self.kx, groups=self.n_bands)
        gy = F.conv2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), self.ky, groups=self.n_bands)
        e = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).mean(dim=1, keepdim=True)
        return torch.sigmoid(e)


class EdgeGuidedRSTBWrapper(nn.Module):
    """Wraps a pretrained RSTB. out = rstb(x) + alpha * edge_gate * (rstb(x) - x)."""
    def __init__(self, rstb):
        super().__init__()
        self.rstb = rstb
        self._edge_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, x_size, edge_gate=None):
        out = self.rstb(x, x_size)
        if edge_gate is not None:
            out = out + self._edge_alpha * edge_gate * (out - x)
        return out


import types

def install_edge_guidance(model, n_bands=4):
    device = next(model.parameters()).device
    model._edge_sobel = SobelEdge(n_bands).to(device)
    model._edge_cache = None

    for i, layer in enumerate(model.layers):
        model.layers[i] = EdgeGuidedRSTBWrapper(layer)

    def patched_forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        if self._edge_cache is not None:
            eg = F.interpolate(self._edge_cache, size=x_size, mode="bilinear", align_corners=False)
            eg_flat = eg.flatten(2).transpose(1, 2)
        else:
            eg_flat = None
        xt = self.patch_embed(x)
        if self.ape:
            xt = xt + self.absolute_pos_embed
        xt = self.pos_drop(xt)
        for layer in self.layers:
            xt = layer(xt, x_size, edge_gate=eg_flat)
        xt = self.norm(xt)
        xt = self.patch_unembed(xt, x_size)
        return xt

    model.forward_features = types.MethodType(patched_forward_features, model)

    orig_forward = model.forward
    def patched_forward(self, x):
        self._edge_cache = self._edge_sobel(x)
        return orig_forward(x)
    model.forward = types.MethodType(patched_forward, model)
    return model


# ---- Band inflation helpers ----
@torch.no_grad()
def inflate_rgb_conv_to_b2b3b4b8(conv_rgb: nn.Conv2d) -> nn.Conv2d:
    assert conv_rgb.in_channels == 3
    w = conv_rgb.weight.data
    out_c, _, kh, kw = w.shape
    new_w = torch.zeros(out_c, 4, kh, kw, dtype=w.dtype, device=w.device)
    R, G, B = w[:, 0], w[:, 1], w[:, 2]
    new_w[:, 0] = B
    new_w[:, 1] = G
    new_w[:, 2] = R
    new_w[:, 3] = R.clone()
    new_conv = nn.Conv2d(4, out_c, (kh, kw), stride=conv_rgb.stride,
                         padding=conv_rgb.padding, bias=(conv_rgb.bias is not None))
    new_conv.weight.data.copy_(new_w)
    if conv_rgb.bias is not None:
        new_conv.bias.data.copy_(conv_rgb.bias.data)
    return new_conv


@torch.no_grad()
def inflate_output_conv_to_4band(conv_rgb: nn.Conv2d) -> nn.Conv2d:
    assert conv_rgb.out_channels == 3
    w = conv_rgb.weight.data
    in_c, kh, kw = w.shape[1:]
    new_w = torch.zeros(4, in_c, kh, kw, dtype=w.dtype, device=w.device)
    R, G, B = w[0], w[1], w[2]
    new_w[0], new_w[1], new_w[2], new_w[3] = B, G, R, R.clone()
    new_conv = nn.Conv2d(in_c, 4, (kh, kw), stride=conv_rgb.stride,
                         padding=conv_rgb.padding, bias=(conv_rgb.bias is not None))
    new_conv.weight.data.copy_(new_w)
    if conv_rgb.bias is not None:
        b = conv_rgb.bias.data
        new_bias = torch.stack([b[2], b[1], b[0], b[0].clone()])
        new_conv.bias.data.copy_(new_bias)
    return new_conv


# ---- Serial FRN (frequency refinement network) ----
def _dct_basis(n=8):
    basis = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        for i in range(n):
            ck = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            basis[k, i] = ck * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return torch.from_numpy(basis)


class BlockDCT(nn.Module):
    def __init__(self, block=8):
        super().__init__()
        self.block = block
        D = _dct_basis(block)
        self.register_buffer("D", D)
        self.register_buffer("Dt", D.t())

    def forward(self, x, inverse=False):
        B, C, H, W = x.shape
        b = self.block
        pad_h = (b - H % b) % b
        pad_w = (b - W % b) % b
        xp = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        Hp, Wp = xp.shape[-2:]
        patches = xp.unfold(2, b, b).unfold(3, b, b)
        nH, nW = patches.shape[2], patches.shape[3]
        patches = patches.reshape(-1, b, b)
        D, Dt = (self.Dt, self.D) if inverse else (self.D, self.Dt)
        out = D @ patches @ Dt
        out = out.reshape(B, C, nH, nW, b, b).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hp, Wp)
        return out[:, :, :H, :W]


class SerialFRN(nn.Module):
    def __init__(self, n_bands=4, feat=24, block=8):
        super().__init__()
        self.dct = BlockDCT(block)
        self.body = nn.Sequential(
            nn.Conv2d(n_bands, feat, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(feat, n_bands, 3, 1, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, sr):
        D = self.dct(sr, inverse=False)
        D_ref = self.body(D)
        refined = self.dct(D_ref, inverse=True)
        return sr + self.alpha * refined


# ---- Combined pipeline module ----
class PipelineB(nn.Module):
    def __init__(self, swinir_model, frn_model):
        super().__init__()
        self.swinir = swinir_model
        self.frn = frn_model

    def forward(self, lr):
        sr0 = self.swinir(lr)
        sr = self.frn(sr0)
        return sr, {"sr_pre_frn": sr0}


# ---------------------------------------------------------------------------
# 3. Quality metrics (same as training notebook)
# ---------------------------------------------------------------------------
def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10((max_val ** 2) / (mse + 1e-10))

def sam(sr, hr, eps=1e-8):
    dot = (sr * hr).sum(dim=1)
    n1 = sr.norm(dim=1); n2 = hr.norm(dim=1)
    cos = (dot / (n1 * n2 + eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)
    return ang.mean() * 180.0 / math.pi

def ndvi(x, red_idx=2, nir_idx=3, eps=1e-6):
    red, nir = x[:, red_idx], x[:, nir_idx]
    return (nir - red) / (nir + red + eps)


# ---------------------------------------------------------------------------
# 4. Image I/O helpers
# ---------------------------------------------------------------------------
SUPPORTED_EXT = {".tif", ".tiff", ".npy", ".pt", ".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return a float32 array of shape [4, H, W]
    with values in [0, 1].
    """
    ext = Path(path).suffix.lower()

    if ext in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError:
            raise ImportError("Install rasterio to load GeoTIFFs: pip install rasterio")
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)  # [C, H, W]
        if img.shape[0] == 4:
            # Assume B2/B3/B4/B8 in correct order; normalize if in DN range
            if img.max() > 10.0:
                img = img / 10000.0
            return img.clip(0, 1)
        elif img.shape[0] >= 3:
            # Take first 3 bands as RGB, duplicate red as NIR
            rgb = img[:3]
            if rgb.max() > 10.0:
                rgb = rgb / 10000.0
            rgb = rgb.clip(0, 1)
            nir = rgb[0:1].copy()  # Red -> NIR proxy
            return np.concatenate([rgb, nir], axis=0)
        else:
            raise ValueError(f"GeoTIFF has {img.shape[0]} bands, expected >=3")

    elif ext == ".npy":
        img = np.load(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 4, axis=0)
        if img.shape[0] != 4:
            raise ValueError(f".npy has shape {img.shape}, expected [4, H, W]")
        if img.max() > 10.0:
            img = img / 10000.0
        return img.clip(0, 1)

    elif ext == ".pt":
        img = torch.load(path, map_location="cpu").numpy().astype(np.float32)
        if img.shape[0] != 4:
            raise ValueError(f".pt tensor has shape {img.shape}, expected [4, H, W]")
        if img.max() > 10.0:
            img = img / 10000.0
        return img.clip(0, 1)

    elif ext in {".png", ".jpg", ".jpeg", ".bmp"}:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0  # [3,H,W]
        # Map: ch0->Blue(B2), ch1->Green(B3), ch2->Red(B4), duplicate Red->NIR(B8)
        b2 = rgb[2:3]  # Blue
        b3 = rgb[1:2]  # Green
        b4 = rgb[0:1]  # Red
        b8 = rgb[0:1].copy()  # NIR <- Red proxy
        return np.concatenate([b2, b3, b4, b8], axis=0)  # [4, H, W]

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def to_rgb(t):
    """Convert a [4, H, W] tensor (B2,B3,B4,B8) to a displayable RGB image."""
    # B4=Red, B3=Green, B2=Blue -> channels [2, 1, 0]
    rgb = t[[2, 1, 0]].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return rgb


# ---------------------------------------------------------------------------
# 5. Tiled inference (for large images)
# ---------------------------------------------------------------------------
@torch.no_grad()
def tiled_inference(model, lr_tensor, tile_size, scale=2, overlap=16, device="cpu"):
    """Run inference in overlapping tiles to fit in GPU memory."""
    _, C, H, W = lr_tensor.shape
    out_H, out_W = H * scale, W * scale
    output = torch.zeros(1, C, out_H, out_W, device="cpu")
    weight_map = torch.zeros(1, 1, out_H, out_W, device="cpu")

    step = tile_size - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = lr_tensor[:, :, y_start:y_end, x_start:x_end].to(device)
            sr_tile, _ = model(tile)
            sr_tile = sr_tile.cpu()

            oy = y_start * scale
            ox = x_start * scale
            oh = sr_tile.shape[2]
            ow = sr_tile.shape[3]
            output[:, :, oy:oy + oh, ox:ox + ow] += sr_tile
            weight_map[:, :, oy:oy + oh, ox:ox + ow] += 1.0

    output /= weight_map.clamp(min=1.0)
    return output


# ---------------------------------------------------------------------------
# 6. Build the full model from checkpoints
# ---------------------------------------------------------------------------
def build_model(bundle_dir, swinir_repo_dir, device, use_frn=True):
    """
    Reconstruct the trained Pipeline B model:
      1. Instantiate SwinIR with the original 3-channel config
      2. Inflate conv_first/conv_last to 4 channels
      3. Install edge-guided modulation wrappers
      4. Load fine-tuned SwinIR checkpoint
      5. Instantiate + load serial FRN
    """
    from models.network_swinir import SwinIR

    bundle = Path(bundle_dir)
    swinir_ckpt = bundle / "swinir_finetuned_best.pt"
    frn_ckpt = bundle / "serial_frn_best.pt"

    if not swinir_ckpt.exists():
        raise FileNotFoundError(f"SwinIR checkpoint not found: {swinir_ckpt}")

    # Read FRN config
    frn_cfg_path = bundle / "frn_config.json"
    if frn_cfg_path.exists():
        with open(frn_cfg_path) as f:
            frn_cfg = json.load(f)
    else:
        frn_cfg = {"block_size": 8, "feat": 24}

    print(f"[INFO] Building SwinIR (classical SR x2, 4-band, edge-guided) ...")

    # Step 1: Instantiate vanilla 3-channel SwinIR
    swinir = SwinIR(
        upscale=2, in_chans=3, img_size=48, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
    ).to(device)

    # Step 2: Inflate 3->4 channels
    swinir.conv_first = inflate_rgb_conv_to_b2b3b4b8(swinir.conv_first).to(device)
    swinir.conv_last = inflate_output_conv_to_4band(swinir.conv_last).to(device)

    # Step 3: Install edge-guided wrappers
    swinir = install_edge_guidance(swinir, n_bands=4)

    # Inflate mean to 4 bands (to avoid dimension mismatch on x - self.mean)
    if hasattr(swinir, "mean") and swinir.mean.shape[1] == 3:
        old_mean = swinir.mean.squeeze()
        # [B2, B3, B4, B8] initialized as [Blue, Green, Red, Red]
        new_mean = torch.tensor([old_mean[2], old_mean[1], old_mean[0], old_mean[0]], device=device).view(1, 4, 1, 1)
        swinir.mean = new_mean

    # Step 4: Load fine-tuned weights
    state = torch.load(str(swinir_ckpt), map_location=device, weights_only=False)
    missing, unexpected = swinir.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded SwinIR checkpoint: "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"       Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"       Unexpected keys (first 5): {unexpected[:5]}")

    # Step 5: Build FRN
    frn = SerialFRN(
        n_bands=4,
        feat=frn_cfg.get("feat", 24),
        block=frn_cfg.get("block_size", 8),
    ).to(device)

    if use_frn and frn_ckpt.exists():
        frn_state = torch.load(str(frn_ckpt), map_location=device, weights_only=False)
        frn.load_state_dict(frn_state)
        print(f"[INFO] Loaded FRN checkpoint: {frn_ckpt.name}")
    elif use_frn:
        print(f"[WARN] FRN checkpoint not found ({frn_ckpt}). FRN will use random weights.")
    else:
        print(f"[INFO] FRN disabled (--no-frn).")

    # Assemble pipeline
    pipeline = PipelineB(swinir, frn).to(device).eval()
    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"[INFO] Pipeline B ready -- {n_params / 1e6:.2f}M parameters total")
    return pipeline


# ---------------------------------------------------------------------------
# 7. Main inference loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Ensure SwinIR source is available
    swinir_repo = ensure_swinir_repo(args.swinir_repo)
    print(f"[INFO] SwinIR source: {swinir_repo}")

    # Locate bundle
    bundle_dir = find_bundle_dir(args.bundle_dir)
    print(f"[INFO] Checkpoint bundle: {bundle_dir}")

    # Build model
    pipeline = build_model(bundle_dir, swinir_repo, device, use_frn=not args.no_frn)

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        )
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    if not files:
        print(f"[ERROR] No supported image files found in {input_path}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} image(s) to process\n")

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import matplotlib for visualisation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Process each image
    results = []
    for idx, fpath in enumerate(files, 1):
        stem = fpath.stem
        print(f"[{idx}/{len(files)}] Processing: {fpath.name}")

        try:
            lr_np = load_image(str(fpath))
        except Exception as e:
            print(f"  x Failed to load: {e}\n")
            continue

        C, H, W = lr_np.shape
        print(f"  Input shape: [{C}, {H}, {W}]")

        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).float()  # [1, 4, H, W]

        # Pad to multiple of window_size (8) to avoid SwinIR shape errors
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            lr_tensor = F.pad(lr_tensor, [0, pad_w, 0, pad_h], mode="reflect")

        t0 = time.time()
        with torch.no_grad():
            if args.tile_size > 0:
                sr_tensor = tiled_inference(
                    pipeline, lr_tensor, args.tile_size, scale=2,
                    overlap=16, device=device
                )
                aux = {}
            else:
                lr_dev = lr_tensor.to(device)
                sr_tensor, aux = pipeline(lr_dev)
                sr_tensor = sr_tensor.cpu()

        elapsed = time.time() - t0

        # Crop away padding from SR output
        sr_H, sr_W = H * 2, W * 2
        sr_tensor = sr_tensor[:, :, :sr_H, :sr_W]

        print(f"  Output shape: [{sr_tensor.shape[1]}, {sr_tensor.shape[2]}, {sr_tensor.shape[3]}]")
        print(f"  Inference time: {elapsed:.2f}s")

        # ---- Save outputs ----
        # 1. Side-by-side comparison PNG
        lr_up = F.interpolate(lr_tensor[:, :, :H, :W], size=(sr_H, sr_W),
                              mode="bicubic", align_corners=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(to_rgb(lr_up[0]))
        axes[0].set_title("Bicubic Upscale (input)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(to_rgb(sr_tensor[0]))
        axes[1].set_title("Terra Lens SR (Pipeline B)", fontsize=12)
        axes[1].axis("off")

        fig.suptitle(f"{fpath.name}  |  {H}x{W} -> {sr_H}x{sr_W}  |  {elapsed:.2f}s",
                     fontsize=11, y=0.02)
        plt.tight_layout()

        comparison_path = out_dir / f"{stem}_comparison.png"
        fig.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  > Saved comparison: {comparison_path}")

        # 2. SR-only output PNG
        sr_only_path = out_dir / f"{stem}_sr.png"
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.imshow(to_rgb(sr_tensor[0]))
        ax2.set_title(f"Terra Lens SR -- {fpath.name}", fontsize=11)
        ax2.axis("off")
        fig2.savefig(str(sr_only_path), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  > Saved SR image:   {sr_only_path}")

        # 3. Optional raw tensor
        if args.save_tensor:
            npy_path = out_dir / f"{stem}_sr.npy"
            np.save(str(npy_path), sr_tensor[0].numpy())
            print(f"  > Saved tensor:     {npy_path}")

        # 4. NDVI comparison (domain-specific diagnostic)
        lr_ndvi = ndvi(lr_tensor[:, :, :H, :W]).squeeze().numpy()
        sr_ndvi = ndvi(sr_tensor).squeeze().numpy()

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes3[0].imshow(lr_ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        axes3[0].set_title("NDVI -- Input (LR)", fontsize=11)
        axes3[0].axis("off")
        plt.colorbar(im0, ax=axes3[0], fraction=0.046)

        im1 = axes3[1].imshow(sr_ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        axes3[1].set_title("NDVI -- Super-Resolved", fontsize=11)
        axes3[1].axis("off")
        plt.colorbar(im1, ax=axes3[1], fraction=0.046)

        ndvi_path = out_dir / f"{stem}_ndvi.png"
        fig3.savefig(str(ndvi_path), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  > Saved NDVI map:   {ndvi_path}")

        results.append({
            "file": fpath.name,
            "input_shape": [C, H, W],
            "output_shape": [4, sr_H, sr_W],
            "inference_time_s": round(elapsed, 3),
        })
        print()

    # Write summary JSON
    summary_path = out_dir / "inference_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump({
            "device": str(device),
            "model": "PipelineB (SwinIR + EdgeGuided + SerialFRN)",
            "scale": 2,
            "frn_enabled": not args.no_frn,
            "results": results,
        }, f, indent=2)
    print(f"[DONE] Processed {len(results)} image(s). Results in: {out_dir}")
    print(f"       Summary: {summary_path}")


if __name__ == "__main__":
    main()
