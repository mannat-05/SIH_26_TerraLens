#!/usr/bin/env python3
"""
Terra Lens — Pipeline B: Sample Evaluation + Raster Visualization
==================================================================

This script:
  1. Loads per-band Sentinel-2 GeoTIFFs from the Sample/ folder
  2. Reconstructs the full Pipeline B model (SwinIR + EdgeGuided + SerialFRN)
  3. Runs x2 super-resolution
  4. Evaluates using Wald's protocol metrics (PSNR, RMSE, SAM, ERGAS, NDVI)
  5. Generates rich raster-based visualizations:
     - True-color & false-color composites (LR vs SR)
     - Per-band comparison panels
     - NDVI & vegetation index maps
     - Spatial profile / cross-section plots
     - Histogram analysis per band
     - Raster metadata inspection

Usage:
  python run_sample_evaluation.py

Requirements:
  pip install torch numpy matplotlib rasterio pillow
  (SwinIR repo will be cloned automatically if not present)

Author: Team TechTonic
"""

import os
import sys
import math
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR / "Sample"
BUNDLE_DIR = SCRIPT_DIR / "terra_lens_pipelineB_bundle (1)"
OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
SCALE = 2

BAND_FILES = {
    "B02": SAMPLE_DIR / "msi_B2_cropped.tif",
    "B03": SAMPLE_DIR / "msi_B3_cropped.tif",
    "B04": SAMPLE_DIR / "msi_B4_cropped.tif",
    "B08": SAMPLE_DIR / "msi_B8_cropped.tif",
}

BAND_LABELS = {
    "B02": "Blue (490 nm)",
    "B03": "Green (560 nm)",
    "B04": "Red (665 nm)",
    "B08": "NIR (842 nm)",
}

BAND_COLORS = {
    "B02": "#3b82f6",
    "B03": "#22c55e",
    "B04": "#ef4444",
    "B08": "#a855f7",
}


# ===================================================================
#  MODEL ARCHITECTURE (exact copy from training notebook)
# ===================================================================
import types


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
    def __init__(self, rstb):
        super().__init__()
        self.rstb = rstb
        self._edge_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, x_size, edge_gate=None):
        out = self.rstb(x, x_size)
        if edge_gate is not None:
            out = out + self._edge_alpha * edge_gate * (out - x)
        return out


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


@torch.no_grad()
def inflate_rgb_conv_to_b2b3b4b8(conv_rgb):
    assert conv_rgb.in_channels == 3
    w = conv_rgb.weight.data
    out_c, _, kh, kw = w.shape
    new_w = torch.zeros(out_c, 4, kh, kw, dtype=w.dtype, device=w.device)
    R, G, B = w[:, 0], w[:, 1], w[:, 2]
    new_w[:, 0], new_w[:, 1], new_w[:, 2], new_w[:, 3] = B, G, R, R.clone()
    new_conv = nn.Conv2d(4, out_c, (kh, kw), stride=conv_rgb.stride,
                         padding=conv_rgb.padding, bias=(conv_rgb.bias is not None))
    new_conv.weight.data.copy_(new_w)
    if conv_rgb.bias is not None:
        new_conv.bias.data.copy_(conv_rgb.bias.data)
    return new_conv


@torch.no_grad()
def inflate_output_conv_to_4band(conv_rgb):
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


class PipelineB(nn.Module):
    def __init__(self, swinir_model, frn_model):
        super().__init__()
        self.swinir = swinir_model
        self.frn = frn_model

    def forward(self, lr):
        sr0 = self.swinir(lr)
        sr = self.frn(sr0)
        return sr, {"sr_pre_frn": sr0}


# ===================================================================
#  WALD'S PROTOCOL DEGRADATION & METRICS
# ===================================================================
def gaussian_kernel2d(kernel_size=9, sigma=1.6, device="cpu"):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).float()


class WaldDegradation(nn.Module):
    """
    Wald's protocol degradation operators:
      B: sensor-MTF-matched Gaussian blur (per-band, depthwise conv)
      S: area subsampling by `scale`
    """
    def __init__(self, n_bands=4, scale=2, kernel_size=9, sigma=1.6):
        super().__init__()
        k = gaussian_kernel2d(kernel_size, sigma)
        self.register_buffer("kernel", k.view(1, 1, kernel_size, kernel_size).repeat(n_bands, 1, 1, 1))
        self.scale = scale
        self.n_bands = n_bands
        self.pad = kernel_size // 2

    def forward(self, x):
        xb = F.conv2d(F.pad(x, [self.pad] * 4, mode="reflect"), self.kernel, groups=self.n_bands)
        xs = xb[:, :, ::self.scale, ::self.scale]
        return xs


def metric_psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10((max_val ** 2) / (mse + 1e-10))


def metric_rmse(sr, hr):
    return torch.sqrt(F.mse_loss(sr, hr) + 1e-12)


def metric_sam(sr, hr, eps=1e-8):
    dot = (sr * hr).sum(dim=1)
    n1 = sr.norm(dim=1)
    n2 = hr.norm(dim=1)
    cos = (dot / (n1 * n2 + eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)
    return ang.mean() * 180.0 / math.pi


def metric_ergas(sr, hr, scale, eps=1e-8):
    n_bands = sr.shape[1]
    total = 0.0
    for b in range(n_bands):
        rmse_b = torch.sqrt(F.mse_loss(sr[:, b], hr[:, b]) + eps)
        mean_b = hr[:, b].mean() + eps
        total += (rmse_b / mean_b) ** 2
    return 100.0 / scale * torch.sqrt(total / n_bands)


def metric_ssim_simple(sr, hr, window_size=11, max_val=1.0):
    """Simplified per-channel SSIM, averaged across bands."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    mu_sr = F.avg_pool2d(sr, window_size, stride=1, padding=window_size // 2)
    mu_hr = F.avg_pool2d(hr, window_size, stride=1, padding=window_size // 2)
    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr
    sigma_sr_sq = F.avg_pool2d(sr ** 2, window_size, stride=1, padding=window_size // 2) - mu_sr_sq
    sigma_hr_sq = F.avg_pool2d(hr ** 2, window_size, stride=1, padding=window_size // 2) - mu_hr_sq
    sigma_sr_hr = F.avg_pool2d(sr * hr, window_size, stride=1, padding=window_size // 2) - mu_sr_hr
    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))
    return ssim_map.mean()


def compute_ndvi(x, red_idx=2, nir_idx=3, eps=1e-6):
    red, nir = x[:, red_idx], x[:, nir_idx]
    return (nir - red) / (nir + red + eps)


def ndvi_consistency(sr, hr, red_idx=2, nir_idx=3):
    return F.l1_loss(compute_ndvi(sr, red_idx, nir_idx),
                     compute_ndvi(hr, red_idx, nir_idx))


# Per-band PSNR
def per_band_psnr(sr, hr, max_val=1.0):
    results = {}
    band_names = ["B02", "B03", "B04", "B08"]
    for i, bname in enumerate(band_names):
        mse = F.mse_loss(sr[:, i], hr[:, i])
        results[bname] = (10 * torch.log10((max_val ** 2) / (mse + 1e-10))).item()
    return results


# ===================================================================
#  RASTER I/O USING RASTERIO
# ===================================================================
def load_sample_bands_rasterio():
    """Load per-band GeoTIFFs and return the stacked array + rasterio metadata."""
    import rasterio

    bands_data = {}
    raster_meta = {}

    for band_name, fpath in BAND_FILES.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Band file not found: {fpath}")
        with rasterio.open(str(fpath)) as src:
            arr = src.read(1).astype(np.float32)
            bands_data[band_name] = arr
            raster_meta[band_name] = {
                "crs": str(src.crs) if src.crs else "N/A",
                "transform": list(src.transform)[:6],
                "bounds": list(src.bounds),
                "resolution": src.res,
                "shape": (src.height, src.width),
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "driver": src.driver,
                "count": src.count,
            }

    # Stack: [B02, B03, B04, B08] -> [4, H, W]
    stack = np.stack([bands_data["B02"], bands_data["B03"],
                      bands_data["B04"], bands_data["B08"]], axis=0)

    return stack, bands_data, raster_meta


def print_raster_metadata(raster_meta):
    """Pretty-print rasterio metadata for all bands."""
    print("\n" + "=" * 70)
    print("  RASTER METADATA (from rasterio)")
    print("=" * 70)
    for band, meta in raster_meta.items():
        print(f"\n  {band} ({BAND_LABELS[band]}):")
        print(f"    CRS:        {meta['crs']}")
        print(f"    Shape:      {meta['shape'][0]} x {meta['shape'][1]} pixels")
        print(f"    Resolution: {meta['resolution'][0]:.2f} x {meta['resolution'][1]:.2f} m")
        print(f"    Bounds:     {meta['bounds']}")
        print(f"    Transform:  {meta['transform']}")
        print(f"    dtype:      {meta['dtype']}")
        print(f"    NoData:     {meta['nodata']}")
        print(f"    Driver:     {meta['driver']}")
    print("=" * 70 + "\n")


# ===================================================================
#  VISUALIZATION FUNCTIONS
# ===================================================================
def normalize_for_display(arr, percentile_low=2, percentile_high=98):
    """Percentile stretch for visual display."""
    lo = np.percentile(arr, percentile_low)
    hi = np.percentile(arr, percentile_high)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def make_rgb_composite(stack, r_idx=2, g_idx=1, b_idx=0):
    """Create an RGB composite from a [4, H, W] stack. Default: true-color (B4, B3, B2)."""
    r = normalize_for_display(stack[r_idx])
    g = normalize_for_display(stack[g_idx])
    b = normalize_for_display(stack[b_idx])
    return np.stack([r, g, b], axis=-1)


def make_false_color_nir(stack):
    """False-color composite: NIR(B08), Red(B04), Green(B03) -> R,G,B."""
    r = normalize_for_display(stack[3])  # NIR
    g = normalize_for_display(stack[2])  # Red
    b = normalize_for_display(stack[1])  # Green
    return np.stack([r, g, b], axis=-1)


def plot_01_raster_overview(stack_raw, raster_meta, out_dir):
    """
    Fig 1: Raster overview — individual bands + true-color + false-color composites.
    Uses rasterio extents for proper geospatial axes.
    """
    import rasterio
    from rasterio.plot import show as rioshow

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)

    band_names = ["B02", "B03", "B04", "B08"]
    cmaps = ["Blues", "Greens", "Reds", "Purples"]

    # Row 1: Individual bands
    for i, (bname, cmap) in enumerate(zip(band_names, cmaps)):
        ax = fig.add_subplot(gs[0, i])
        band = stack_raw[i]
        im = ax.imshow(normalize_for_display(band), cmap=cmap, aspect="auto")
        ax.set_title(f"{bname}\n{BAND_LABELS[bname]}", fontsize=11, fontweight="bold",
                     color=BAND_COLORS[bname])
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label="Normalized")

    # Row 2: Composites
    ax_tc = fig.add_subplot(gs[1, 0:2])
    tc = make_rgb_composite(stack_raw)
    ax_tc.imshow(tc, aspect="auto")
    ax_tc.set_title("True Color Composite\n(B4-Red, B3-Green, B2-Blue)", fontsize=12, fontweight="bold")
    ax_tc.set_xlabel("Column"); ax_tc.set_ylabel("Row")

    ax_fc = fig.add_subplot(gs[1, 2:4])
    fc = make_false_color_nir(stack_raw)
    ax_fc.imshow(fc, aspect="auto")
    ax_fc.set_title("False Color Composite\n(B8-NIR, B4-Red, B3-Green)", fontsize=12, fontweight="bold")
    ax_fc.set_xlabel("Column"); ax_fc.set_ylabel("Row")

    # Add CRS info
    meta_b02 = raster_meta.get("B02", {})
    crs_str = meta_b02.get("crs", "N/A")
    res = meta_b02.get("resolution", ("?", "?"))
    fig.suptitle(f"MSI Raster Overview  |  CRS: {crs_str}  |  Resolution: {res[0]}m x {res[1]}m",
                 fontsize=14, fontweight="bold", y=0.98)

    path = out_dir / "01_raster_overview.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_02_band_histograms(stack_raw, out_dir):
    """Fig 2: Per-band histograms showing reflectance distribution."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    band_names = ["B02", "B03", "B04", "B08"]

    for i, (ax, bname) in enumerate(zip(axes, band_names)):
        data = stack_raw[i].flatten()
        valid = data[data > 0]  # exclude nodata/zeros
        ax.hist(valid, bins=100, color=BAND_COLORS[bname], alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.mean(valid), color="black", linestyle="--", linewidth=1.2, label=f"Mean: {np.mean(valid):.1f}")
        ax.axvline(np.median(valid), color="gray", linestyle=":", linewidth=1.2, label=f"Median: {np.median(valid):.1f}")
        ax.set_title(f"{bname} — {BAND_LABELS[bname]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Reflectance (DN)")
        ax.set_ylabel("Pixel Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Band Reflectance Histograms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "02_band_histograms.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_03_sr_comparison(lr_stack_norm, sr_stack_norm, bicubic_stack_norm, out_dir):
    """Fig 3: Side-by-side: Bicubic vs Terra Lens SR (true-color + false-color)."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    # True color
    lr_tc = make_rgb_composite(lr_stack_norm)
    bic_tc = make_rgb_composite(bicubic_stack_norm)
    sr_tc = make_rgb_composite(sr_stack_norm)

    axes[0, 0].imshow(lr_tc); axes[0, 0].set_title("LR Input\n(Original)", fontsize=12, fontweight="bold")
    axes[0, 1].imshow(bic_tc); axes[0, 1].set_title("Bicubic Upscale\n(Baseline)", fontsize=12, fontweight="bold")
    axes[0, 2].imshow(sr_tc); axes[0, 2].set_title("Terra Lens SR\n(Pipeline B)", fontsize=12, fontweight="bold", color="#059669")

    # False color
    lr_fc = make_false_color_nir(lr_stack_norm)
    bic_fc = make_false_color_nir(bicubic_stack_norm)
    sr_fc = make_false_color_nir(sr_stack_norm)

    axes[1, 0].imshow(lr_fc); axes[1, 0].set_title("LR — False Color (NIR,R,G)", fontsize=11)
    axes[1, 1].imshow(bic_fc); axes[1, 1].set_title("Bicubic — False Color", fontsize=11)
    axes[1, 2].imshow(sr_fc); axes[1, 2].set_title("SR — False Color", fontsize=11, color="#059669")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("Super-Resolution Comparison: True Color & False Color Composites", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "03_sr_comparison_composites.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_04_per_band_sr(lr_stack_norm, sr_stack_norm, out_dir):
    """Fig 4: Per-band LR vs SR comparison (4 bands x 2 columns)."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    band_names = ["B02", "B03", "B04", "B08"]
    cmaps = ["Blues", "Greens", "Reds", "Purples"]

    for i, (bname, cmap) in enumerate(zip(band_names, cmaps)):
        lr_band = normalize_for_display(lr_stack_norm[i])
        sr_band = normalize_for_display(sr_stack_norm[i])

        # Upscale LR for visual comparison
        lr_up = np.repeat(np.repeat(lr_band, SCALE, axis=0), SCALE, axis=1)
        h_min = min(lr_up.shape[0], sr_band.shape[0])
        w_min = min(lr_up.shape[1], sr_band.shape[1])
        lr_up = lr_up[:h_min, :w_min]
        sr_crop = sr_band[:h_min, :w_min]

        axes[i, 0].imshow(lr_up, cmap=cmap, aspect="auto")
        axes[i, 0].set_title(f"{bname} — LR (upscaled nearest)", fontsize=11, color=BAND_COLORS[bname])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sr_crop, cmap=cmap, aspect="auto")
        axes[i, 1].set_title(f"{bname} — SR (Terra Lens)", fontsize=11, fontweight="bold", color=BAND_COLORS[bname])
        axes[i, 1].axis("off")

        # Difference map
        diff_map = np.abs(sr_crop.astype(float) - lr_up.astype(float))
        im_diff = axes[i, 2].imshow(diff_map, cmap="hot", aspect="auto", vmin=0, vmax=0.3)
        axes[i, 2].set_title(f"{bname} — |SR - LR| Difference", fontsize=11)
        axes[i, 2].axis("off")
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_diff, cax=cax)

    fig.suptitle("Per-Band Super-Resolution: LR vs SR vs Difference", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "04_per_band_sr_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_05_ndvi_maps(lr_stack_norm, sr_stack_norm, out_dir):
    """Fig 5: NDVI maps — LR vs SR, plus difference."""
    lr_t = torch.from_numpy(lr_stack_norm).unsqueeze(0)
    sr_t = torch.from_numpy(sr_stack_norm).unsqueeze(0)

    lr_ndvi = compute_ndvi(lr_t).squeeze().numpy()
    sr_ndvi = compute_ndvi(sr_t).squeeze().numpy()

    # Upsample LR NDVI to SR size for difference
    lr_ndvi_up = np.repeat(np.repeat(lr_ndvi, SCALE, axis=0), SCALE, axis=1)
    h_min = min(lr_ndvi_up.shape[0], sr_ndvi.shape[0])
    w_min = min(lr_ndvi_up.shape[1], sr_ndvi.shape[1])
    diff_ndvi = sr_ndvi[:h_min, :w_min] - lr_ndvi_up[:h_min, :w_min]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    im0 = axes[0].imshow(lr_ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
    axes[0].set_title("NDVI — Low Resolution Input", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="NDVI")

    im1 = axes[1].imshow(sr_ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
    axes[1].set_title("NDVI — Super-Resolved (Terra Lens)", fontsize=12, fontweight="bold", color="#059669")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="NDVI")

    im2 = axes[2].imshow(diff_ndvi, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
    axes[2].set_title("NDVI Difference (SR - LR)", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Delta NDVI")

    fig.suptitle("Normalized Difference Vegetation Index (NDVI) Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "05_ndvi_analysis.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_06_spatial_profile(lr_stack_norm, sr_stack_norm, out_dir):
    """Fig 6: Spatial profile / cross-section across a horizontal line for each band."""
    H_lr = lr_stack_norm.shape[1]
    H_sr = sr_stack_norm.shape[1]
    profile_row_lr = H_lr // 2
    profile_row_sr = H_sr // 2

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    band_names = ["B02", "B03", "B04", "B08"]

    for i, (ax, bname) in enumerate(zip(axes.flat, band_names)):
        lr_profile = lr_stack_norm[i, profile_row_lr, :]
        sr_profile = sr_stack_norm[i, profile_row_sr, :]

        # Upsample LR profile for direct comparison
        lr_x = np.arange(len(lr_profile)) * SCALE
        sr_x = np.arange(len(sr_profile))

        ax.plot(lr_x, lr_profile, '-', color=BAND_COLORS[bname], alpha=0.5,
                linewidth=1.5, label=f"LR (row {profile_row_lr})")
        ax.plot(sr_x[:len(sr_profile)], sr_profile, '-', color=BAND_COLORS[bname],
                linewidth=0.8, alpha=0.9, label=f"SR (row {profile_row_sr})")
        ax.set_title(f"{bname} — {BAND_LABELS[bname]}", fontsize=12, fontweight="bold",
                     color=BAND_COLORS[bname])
        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Reflectance (normalized)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Spatial Cross-Section Profiles (middle row)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "06_spatial_profiles.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_07_wald_metrics(wald_metrics, per_band_metrics, out_dir):
    """Fig 7: Wald's protocol metrics dashboard."""
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.3)

    # Left: Overall metrics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    metric_names = list(wald_metrics.keys())
    metric_vals = list(wald_metrics.values())
    colors_bar = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    bars = ax1.barh(metric_names, metric_vals, color=colors_bar[:len(metric_names)],
                    edgecolor="white", linewidth=1.5, height=0.6)
    for bar, val in zip(bars, metric_vals):
        ax1.text(bar.get_width() + 0.01 * max(abs(v) for v in metric_vals), bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
    ax1.set_title("Wald Protocol Metrics\n(Overall)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Value")
    ax1.grid(True, axis="x", alpha=0.3)

    # Right: Per-band PSNR
    ax2 = fig.add_subplot(gs[0, 1])
    bands = list(per_band_metrics.keys())
    psnr_vals = list(per_band_metrics.values())
    bar_colors = [BAND_COLORS[b] for b in bands]

    bars2 = ax2.bar(bands, psnr_vals, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars2, psnr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Band PSNR (dB)\n(Higher is better)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Wald's Protocol Evaluation Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "07_wald_metrics_dashboard.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


def plot_08_edge_enhancement(lr_stack_norm, sr_stack_norm, out_dir):
    """Fig 8: Edge maps (Sobel) of LR vs SR on the Red band to show detail recovery."""
    from scipy.ndimage import sobel as scipy_sobel

    lr_red = normalize_for_display(lr_stack_norm[2])
    sr_red = normalize_for_display(sr_stack_norm[2])

    lr_up = np.repeat(np.repeat(lr_red, SCALE, axis=0), SCALE, axis=1)
    h_min = min(lr_up.shape[0], sr_red.shape[0])
    w_min = min(lr_up.shape[1], sr_red.shape[1])
    lr_up = lr_up[:h_min, :w_min]
    sr_red = sr_red[:h_min, :w_min]

    lr_edge = np.sqrt(scipy_sobel(lr_up, axis=0)**2 + scipy_sobel(lr_up, axis=1)**2)
    sr_edge = np.sqrt(scipy_sobel(sr_red, axis=0)**2 + scipy_sobel(sr_red, axis=1)**2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    axes[0].imshow(lr_edge, cmap="inferno", aspect="auto")
    axes[0].set_title("Edge Map — LR (upscaled nearest)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sr_edge, cmap="inferno", aspect="auto")
    axes[1].set_title("Edge Map — SR (Terra Lens)", fontsize=12, fontweight="bold", color="#059669")
    axes[1].axis("off")

    edge_diff = sr_edge - lr_edge
    im = axes[2].imshow(edge_diff, cmap="RdBu_r", aspect="auto",
                        vmin=-np.percentile(np.abs(edge_diff), 95),
                        vmax=np.percentile(np.abs(edge_diff), 95))
    axes[2].set_title("Edge Enhancement (SR - LR)", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("Edge Detection Analysis (Red Band / B04)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "08_edge_enhancement.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [VIZ] Saved: {path.name}")
    return path


# ===================================================================
#  MODEL LOADING
# ===================================================================
def ensure_swinir_repo():
    candidates = [
        SCRIPT_DIR / "SwinIR",
        Path.cwd() / "SwinIR",
    ]
    for c in candidates:
        if (c / "models" / "network_swinir.py").exists():
            sys.path.insert(0, str(c))
            return str(c)
    repo_dir = SCRIPT_DIR / "SwinIR"
    print("[INFO] Cloning official SwinIR repo ...")
    ret = os.system(f'git clone -q https://github.com/JingyunLiang/SwinIR.git "{repo_dir}"')
    if ret != 0:
        raise RuntimeError("Failed to clone SwinIR. Clone it manually.")
    sys.path.insert(0, str(repo_dir))
    return str(repo_dir)


def build_pipeline(device):
    from models.network_swinir import SwinIR

    swinir_ckpt = BUNDLE_DIR / "swinir_finetuned_best.pt"
    frn_ckpt = BUNDLE_DIR / "serial_frn_best.pt"

    frn_cfg_path = BUNDLE_DIR / "frn_config.json"
    frn_cfg = {"block_size": 8, "feat": 24}
    if frn_cfg_path.exists():
        with open(frn_cfg_path) as f:
            frn_cfg = json.load(f)

    print("[INFO] Building SwinIR (classical SR x2, 4-band, edge-guided) ...")
    swinir = SwinIR(
        upscale=2, in_chans=3, img_size=48, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
    ).to(device)

    swinir.conv_first = inflate_rgb_conv_to_b2b3b4b8(swinir.conv_first).to(device)
    swinir.conv_last = inflate_output_conv_to_4band(swinir.conv_last).to(device)
    swinir = install_edge_guidance(swinir, n_bands=4)

    # Inflate mean to 4 bands (to avoid dimension mismatch on x - self.mean)
    if hasattr(swinir, "mean") and swinir.mean.shape[1] == 3:
        old_mean = swinir.mean.squeeze()
        # [B2, B3, B4, B8] initialized as [Blue, Green, Red, Red]
        new_mean = torch.tensor([old_mean[2], old_mean[1], old_mean[0], old_mean[0]], device=device).view(1, 4, 1, 1)
        swinir.mean = new_mean

    state = torch.load(str(swinir_ckpt), map_location=device, weights_only=False)
    missing, unexpected = swinir.load_state_dict(state, strict=False)
    print(f"[INFO] SwinIR loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    frn = SerialFRN(
        n_bands=4,
        feat=frn_cfg.get("feat", 24),
        block=frn_cfg.get("block_size", 8),
    ).to(device)

    if frn_ckpt.exists():
        frn_state = torch.load(str(frn_ckpt), map_location=device, weights_only=False)
        frn.load_state_dict(frn_state)
        print(f"[INFO] FRN loaded: {frn_ckpt.name}")

    pipeline = PipelineB(swinir, frn).to(device).eval()
    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"[INFO] Pipeline B ready -- {n_params / 1e6:.2f}M parameters")
    return pipeline


# ===================================================================
#  TILED INFERENCE (for large images)
# ===================================================================
@torch.no_grad()
def tiled_inference(model, lr_tensor, tile_size=128, scale=2, overlap=16, device="cpu"):
    _, C, H, W = lr_tensor.shape
    out_H, out_W = H * scale, W * scale
    output = torch.zeros(1, C, out_H, out_W, device="cpu")
    weight_map = torch.zeros(1, 1, out_H, out_W, device="cpu")

    step = tile_size - overlap
    n_tiles = 0
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
            n_tiles += 1

    output /= weight_map.clamp(min=1.0)
    print(f"  [INFO] Tiled inference: {n_tiles} tiles processed")
    return output


# ===================================================================
#  MAIN
# ===================================================================
def main():
    print("=" * 70)
    print("  TERRA LENS — Pipeline B Evaluation & Raster Visualization")
    print("  Team TechTonic")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Load sample MSI data with rasterio
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading MSI bands with rasterio ...")
    stack_raw, bands_data, raster_meta = load_sample_bands_rasterio()
    print(f"  Stacked shape: {stack_raw.shape}")
    print(f"  Value range: [{stack_raw.min():.2f}, {stack_raw.max():.2f}]")

    print_raster_metadata(raster_meta)

    # Normalize to [0, 1]
    if stack_raw.max() > 10.0:
        stack_norm = (stack_raw / 10000.0).clip(0, 1).astype(np.float32)
    elif stack_raw.max() > 1.5:
        stack_norm = (stack_raw / stack_raw.max()).clip(0, 1).astype(np.float32)
    else:
        stack_norm = stack_raw.clip(0, 1).astype(np.float32)

    print(f"  Normalized range: [{stack_norm.min():.4f}, {stack_norm.max():.4f}]")

    # ------------------------------------------------------------------
    # Step 2: Raster visualizations of the RAW input
    # ------------------------------------------------------------------
    print("\n[STEP 2] Generating raster visualizations of input MSI data ...")
    plot_01_raster_overview(stack_raw, raster_meta, OUTPUT_DIR)
    plot_02_band_histograms(stack_raw, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Step 3: Build and load the model
    # ------------------------------------------------------------------
    print("\n[STEP 3] Building Pipeline B model ...")
    swinir_repo = ensure_swinir_repo()
    print(f"  SwinIR source: {swinir_repo}")
    pipeline = build_pipeline(device)

    # ------------------------------------------------------------------
    # Step 4: Run inference
    # ------------------------------------------------------------------
    print("\n[STEP 4] Running super-resolution inference ...")
    C, H, W = stack_norm.shape
    lr_tensor = torch.from_numpy(stack_norm).unsqueeze(0).float()

    # Pad to multiple of 8
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        lr_tensor = F.pad(lr_tensor, [0, pad_w, 0, pad_h], mode="reflect")
        print(f"  Padded: [{H},{W}] -> [{lr_tensor.shape[2]},{lr_tensor.shape[3]}]")

    t0 = time.time()

    # Use tiled inference if image is large (> 512px in any dim)
    if H > 512 or W > 512:
        print(f"  Large image detected ({H}x{W}), using tiled inference (tile=192, overlap=24) ...")
        sr_tensor = tiled_inference(pipeline, lr_tensor, tile_size=192, scale=SCALE,
                                    overlap=24, device=device)
    else:
        with torch.no_grad():
            lr_dev = lr_tensor.to(device)
            sr_tensor, aux = pipeline(lr_dev)
            sr_tensor = sr_tensor.cpu()

    elapsed = time.time() - t0

    # Crop to exact output size
    sr_H, sr_W = H * SCALE, W * SCALE
    sr_tensor = sr_tensor[:, :, :sr_H, :sr_W].clamp(0, 1)
    sr_stack_norm = sr_tensor.squeeze(0).numpy()

    # Also make bicubic baseline
    bicubic_tensor = F.interpolate(
        torch.from_numpy(stack_norm).unsqueeze(0),
        size=(sr_H, sr_W), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    bicubic_stack_norm = bicubic_tensor.squeeze(0).numpy()

    print(f"  Input shape:  [{C}, {H}, {W}]")
    print(f"  Output shape: [4, {sr_H}, {sr_W}]")
    print(f"  Inference time: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Wald's Protocol Evaluation
    # ------------------------------------------------------------------
    print("\n[STEP 5] Running Wald's protocol evaluation ...")
    print("  (The HR reference = original input; LR = degraded original)")
    print("  (SR is produced from the degraded LR; compared against original as GT)")

    wald_deg = WaldDegradation(n_bands=4, scale=SCALE).to(device)

    # Wald's protocol: use the *original* as HR ground-truth, degrade it to get LR_sim,
    # super-resolve LR_sim, compare SR vs original.
    hr_wald = torch.from_numpy(stack_norm).unsqueeze(0).float().to(device)
    with torch.no_grad():
        lr_sim = wald_deg(hr_wald)   # Simulated LR

    # Pad simulated LR
    _, _, h_sim, w_sim = lr_sim.shape
    pad_h2 = (8 - h_sim % 8) % 8
    pad_w2 = (8 - w_sim % 8) % 8
    lr_sim_padded = lr_sim
    if pad_h2 > 0 or pad_w2 > 0:
        lr_sim_padded = F.pad(lr_sim, [0, pad_w2, 0, pad_h2], mode="reflect")

    # Run SR on simulated LR
    with torch.no_grad():
        if h_sim > 512 or w_sim > 512:
            sr_wald = tiled_inference(pipeline, lr_sim_padded.cpu(), tile_size=192,
                                      scale=SCALE, overlap=24, device=device)
        else:
            sr_wald_out, _ = pipeline(lr_sim_padded.to(device))
            sr_wald = sr_wald_out.cpu()

    sr_wald = sr_wald[:, :, :h_sim * SCALE, :w_sim * SCALE].clamp(0, 1)

    # Make bicubic baseline from simulated LR
    bicubic_wald = F.interpolate(lr_sim.cpu(), size=(h_sim * SCALE, w_sim * SCALE),
                                  mode="bicubic", align_corners=False).clamp(0, 1)

    # Ensure shapes match for comparison (crop to min dimensions)
    hr_compare = hr_wald.cpu()
    sr_compare = sr_wald
    bic_compare = bicubic_wald

    min_h = min(hr_compare.shape[2], sr_compare.shape[2], bic_compare.shape[2])
    min_w = min(hr_compare.shape[3], sr_compare.shape[3], bic_compare.shape[3])
    hr_compare = hr_compare[:, :, :min_h, :min_w]
    sr_compare = sr_compare[:, :, :min_h, :min_w]
    bic_compare = bic_compare[:, :, :min_h, :min_w]

    # Compute metrics: Terra Lens SR vs HR
    print("\n  --- Terra Lens SR vs Ground Truth ---")
    wald_results = {
        "PSNR (dB)": metric_psnr(sr_compare, hr_compare).item(),
        "RMSE": metric_rmse(sr_compare, hr_compare).item(),
        "SAM (deg)": metric_sam(sr_compare, hr_compare).item(),
        "ERGAS": metric_ergas(sr_compare, hr_compare, SCALE).item(),
        "SSIM": metric_ssim_simple(sr_compare, hr_compare).item(),
        "NDVI MAE": ndvi_consistency(sr_compare, hr_compare).item(),
    }
    for k, v in wald_results.items():
        print(f"    {k:12s}: {v:.6f}")

    pb_psnr = per_band_psnr(sr_compare, hr_compare)
    print("\n  Per-band PSNR:")
    for b, v in pb_psnr.items():
        print(f"    {b}: {v:.4f} dB")

    # Compute metrics: Bicubic baseline vs HR
    print("\n  --- Bicubic Baseline vs Ground Truth ---")
    bic_results = {
        "PSNR (dB)": metric_psnr(bic_compare, hr_compare).item(),
        "RMSE": metric_rmse(bic_compare, hr_compare).item(),
        "SAM (deg)": metric_sam(bic_compare, hr_compare).item(),
        "ERGAS": metric_ergas(bic_compare, hr_compare, SCALE).item(),
        "SSIM": metric_ssim_simple(bic_compare, hr_compare).item(),
        "NDVI MAE": ndvi_consistency(bic_compare, hr_compare).item(),
    }
    print("  Bicubic baseline metrics:")
    for k, v in bic_results.items():
        print(f"    {k:12s}: {v:.6f}")

    bic_pb_psnr = per_band_psnr(bic_compare, hr_compare)

    # ------------------------------------------------------------------
    # Step 6: Generate all visualizations
    # ------------------------------------------------------------------
    print("\n[STEP 6] Generating visualizations ...")
    plot_03_sr_comparison(stack_norm, sr_stack_norm, bicubic_stack_norm, OUTPUT_DIR)
    plot_04_per_band_sr(stack_norm, sr_stack_norm, OUTPUT_DIR)
    plot_05_ndvi_maps(stack_norm, sr_stack_norm, OUTPUT_DIR)
    plot_06_spatial_profile(stack_norm, sr_stack_norm, OUTPUT_DIR)
    plot_07_wald_metrics(wald_results, pb_psnr, OUTPUT_DIR)

    # Edge enhancement visualization (try scipy, skip if not available)
    try:
        from scipy.ndimage import sobel as scipy_sobel
        plot_08_edge_enhancement(stack_norm, sr_stack_norm, OUTPUT_DIR)
    except ImportError:
        print("  [SKIP] scipy not installed, skipping edge enhancement plot")

    # ------------------------------------------------------------------
    # Step 7: Save comprehensive results JSON
    # ------------------------------------------------------------------
    print("\n[STEP 7] Saving results ...")
    all_results = {
        "device": str(device),
        "model": "PipelineB (SwinIR + EdgeGuided + SerialFRN)",
        "scale": SCALE,
        "input_shape": [C, H, W],
        "sr_output_shape": [4, sr_H, sr_W],
        "inference_time_s": round(elapsed, 3),
        "wald_protocol": {
            "description": "SR produced from Wald-degraded input, compared against original as GT",
            "terra_lens_sr": wald_results,
            "bicubic_baseline": bic_results,
            "improvement": {k: wald_results[k] - bic_results[k] for k in wald_results},
        },
        "per_band_psnr": {
            "terra_lens_sr": pb_psnr,
            "bicubic_baseline": bic_pb_psnr,
        },
        "raster_metadata": {b: {k: str(v) for k, v in meta.items()} for b, meta in raster_meta.items()},
    }

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.1f} KB)")
    print()

    print("  Wald Protocol Summary (Terra Lens vs Bicubic):")
    print(f"    {'Metric':<15s} {'Terra Lens':>12s} {'Bicubic':>12s} {'Delta':>12s}")
    print(f"    {'─' * 51}")
    for k in wald_results:
        delta = wald_results[k] - bic_results[k]
        better = "+" if ((k == "PSNR (dB)" or k == "SSIM") and delta > 0) or \
                         ((k not in ["PSNR (dB)", "SSIM"]) and delta < 0) else "-"
        print(f"    {k:<15s} {wald_results[k]:>12.4f} {bic_results[k]:>12.4f} {delta:>+12.4f} {better}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
