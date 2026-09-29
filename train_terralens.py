import os

# Must be set BEFORE torch initialises CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import time
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader, Subset

# ----------------------------------------------------------------------
# Config (override with environment variables if you like)
# ----------------------------------------------------------------------
SEED = 42
DATA_DIR = os.environ.get("DATA_DIR", "./s2_naip_dataset")
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", 128))
SCALE = 4
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", 2))      # effective batch = 32
NUM_WORKERS = int(os.environ.get(
    "NUM_WORKERS", max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 2)) - 1)))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
AMP_DTYPE = torch.bfloat16 if (USE_CUDA and torch.cuda.is_bf16_supported()) else torch.float16
# GradScaler is only required for fp16, not bf16
USE_SCALER = USE_CUDA and AMP_DTYPE == torch.float16


# ----------------------------------------------------------------------
# Early stopping (also writes best weights to disk)
# ----------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, mode='max', verbose=True, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def _snapshot(self, model):
        self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.save_path:
            torch.save(self.best_weights, self.save_path)

    def set_baseline(self, score, model):
        """Use an already-achieved score (e.g. end of previous stage) as the bar to beat."""
        self.best_score = score
        self._snapshot(model)

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self._snapshot(model)
            return True

        if self.mode == 'max':
            improved = (current_score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_score) > self.min_delta

        if improved:
            if self.verbose:
                print(f"  [EarlyStopping] Improved ({self.best_score:.4f} -> {current_score:.4f}). Checkpoint saved.")
            self.best_score = current_score
            self._snapshot(model)
            self.counter = 0
            return True

        self.counter += 1
        if self.verbose:
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in self.best_weights.items()})


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
IMG_EXTS = (".npy", ".tif", ".tiff", ".png")


def find_files(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset directory '{root}' does not exist. Download/unzip the Kaggle dataset first "
            f"(see run_terralens.sh).")
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No {IMG_EXTS} files found under '{root}'. Check the folder layout with: find {root} | head")
    # If the dataset has separate NAIP (high-res) and Sentinel-2 folders, use NAIP as HR target.
    naip = [f for f in files if "naip" in os.path.relpath(f, root).lower()]
    return naip if naip else files


def read_image(path):
    """Return float32 array (4, H, W) in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        arr = np.array(Image.open(path))

    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:            # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
    elif arr.shape[0] != 4:
        raise ValueError(f"{path}: expected 4 bands (RGB+NIR), got array shape {arr.shape}")

    # Normalise by dtype (deterministic, not per-image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 10000.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 10000.0
    return np.clip(arr, 0.0, 1.0)


class S2NAIPDataset(Dataset):
    def __init__(self, files, patch_size=128, scale=4, is_train=True):
        self.files = files
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = read_image(self.files[idx])
        _, h, w = hr.shape
        ps = self.patch_size

        if h < ps or w < ps:
            hr = np.pad(hr, ((0, 0), (0, max(0, ps - h)), (0, max(0, ps - w))), mode="edge")
            _, h, w = hr.shape

        if self.is_train:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
        else:  # deterministic centre crop for validation
            y = (h - ps) // 2
            x = (w - ps) // 2
        hr_t = torch.from_numpy(np.ascontiguousarray(hr[:, y:y + ps, x:x + ps])).float()

        lr_t = F.interpolate(hr_t.unsqueeze(0), scale_factor=1.0 / self.scale,
                             mode='bicubic', align_corners=False, antialias=True).squeeze(0).clamp(0, 1)
        return lr_t, hr_t


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class SobelEdge(nn.Module):
    """Per-image-normalised Sobel gradient magnitude in [0, 1), shape (B, 1, H, W)."""
    def __init__(self, n_bands=4):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1))
        self.n_bands = n_bands

    @torch.no_grad()
    def forward(self, x):
        x = x.float()
        xp = F.pad(x, [1, 1, 1, 1], mode="reflect")
        gx = F.conv2d(xp, self.kx, groups=self.n_bands)
        gy = F.conv2d(xp, self.ky, groups=self.n_bands)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).mean(dim=1, keepdim=True)
        mag = mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-6)   # 1.0 == average edge strength
        return torch.tanh(mag)


class RSTB(nn.Module):
    """Global-attention block guided by a Sobel edge map.
    (a) edge embedding is added to the features, (b) attention logits get a learnable bias
    toward edge-rich key tokens. Both are zero-initialised, so training starts from the plain block."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.edge_embed = nn.Conv2d(1, dim, 1)
        nn.init.zeros_(self.edge_embed.weight)
        nn.init.zeros_(self.edge_embed.bias)
        self.edge_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_map):
        B, C, H, W = x.shape
        x = x + self.edge_embed(edge_map)                    # (a) edge-aware features
        x_flat = x.flatten(2).transpose(1, 2)                # (B, N, C)
        e_key = edge_map.flatten(2)                          # (B, 1, N), broadcast over queries
        norm_x = self.norm(x_flat)
        q, k, v = torch.chunk(self.qkv(norm_x), 3, dim=-1)
        logits = (q @ k.transpose(-2, -1)) * (C ** -0.5) + self.edge_beta * e_key   # (b) edge-biased attention
        attn = torch.softmax(logits.float(), dim=-1)
        out = self.proj(attn.to(v.dtype) @ v) + x_flat
        out = out + self.mlp(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


class FRN(nn.Module):
    def __init__(self, n_bands=4, feat=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_bands * 2, feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, n_bands * 2, 3, 1, 1)
        )

    def forward(self, x):
        # FFT / complex ops do NOT support bfloat16 -> run this block in float32, outside autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            fft_x = torch.fft.rfft2(x, norm="backward")
            fft_cat = torch.cat([fft_x.real, fft_x.imag], dim=1)
            out_fft = self.body(fft_cat)
            real, imag = torch.chunk(out_fft, 2, dim=1)
            spatial = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm="backward")
            ac_energy = (out_fft ** 2).mean(dim=1, keepdim=True)
        return spatial, ac_energy


class TerraLens(nn.Module):
    def __init__(self, n_bands=4, dim=48, scale=4, use_checkpointing=True):
        super().__init__()
        self.scale = scale
        self.use_checkpointing = use_checkpointing
        self.edge = SobelEdge(n_bands)
        self.shallow = nn.Conv2d(n_bands, dim, 3, 1, 1)
        self.rstb = RSTB(dim)
        self.spatial_out = nn.Conv2d(dim, n_bands, 3, 1, 1)
        self.frn = FRN(n_bands)
        self.gate = nn.Sequential(nn.Conv2d(1, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_bands, dim * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, n_bands, 3, 1, 1)
        )

    def forward(self, lr):
        edge_map = self.edge(lr)
        shallow_feat = self.shallow(lr)
        if self.use_checkpointing and self.training:
            feat = checkpoint(self.rstb, shallow_feat, edge_map, use_reentrant=False)
        else:
            feat = self.rstb(shallow_feat, edge_map)

        x_spatial = self.spatial_out(feat)
        x_freq, ac_energy = self.frn(lr)
        g = torch.sigmoid(self.gate(ac_energy))
        fused = g * x_freq + (1 - g) * x_spatial
        return self.upsampler(fused) + F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)


def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr.float(), hr.float())
    return 10 * torch.log10(1.0 / (mse + 1e-10))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
@torch.no_grad()
def bicubic_baseline_psnr(loader):
    tot = 0.0
    for lr_v, hr_v in loader:
        up = F.interpolate(lr_v.to(DEVICE), scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)
        tot += calculate_psnr(up, hr_v.to(DEVICE)).item()
    return tot / max(1, len(loader))


def run_stage(model, train_loader, val_loader, optimizer, scaler, max_epochs, stage_name,
              accum_steps=2, patience=10, baseline=None, save_path=None):
    print(f"\n--- Starting {stage_name} (Max Epochs: {max_epochs}, EarlyStop Patience: {patience}) ---", flush=True)
    early_stopper = EarlyStopping(patience=patience, min_delta=0.01, mode='max', verbose=True, save_path=save_path)
    if baseline is not None:
        early_stopper.set_baseline(baseline, model)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (lr_b, hr_b) in enumerate(train_loader):
            lr_b = lr_b.to(DEVICE, non_blocking=True)
            hr_b = hr_b.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_CUDA):
                sr_b = model(lr_b)
                loss = F.l1_loss(sr_b.float(), hr_b) / accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if USE_CUDA:
            torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v = lr_v.to(DEVICE, non_blocking=True)
                hr_v = hr_v.to(DEVICE, non_blocking=True)
                val_psnr += calculate_psnr(model(lr_v).clamp(0, 1), hr_v).item()

        avg_val_psnr = val_psnr / max(1, len(val_loader))
        print(f"[{stage_name} | Epoch {epoch:02d}/{max_epochs:02d}] "
              f"Loss: {train_loss / max(1, len(train_loader)):.4f} | Val PSNR: {avg_val_psnr:.2f} dB | "
              f"Time: {time.time() - t0:.1f}s", flush=True)

        early_stopper(avg_val_psnr, model)
        if early_stopper.early_stop:
            print(f"--> Early stopping triggered for {stage_name} at epoch {epoch}.", flush=True)
            break

    early_stopper.restore_best_weights(model)
    return early_stopper.best_score


def main():
    if not USE_CUDA and os.environ.get("ALLOW_CPU", "0") != "1":
        raise RuntimeError("CUDA not available. Check --gres / module load cuda / PyTorch build. "
                           "Set ALLOW_CPU=1 to force CPU training.")
    os.makedirs(CKPT_DIR, exist_ok=True)

    files = find_files(DATA_DIR)
    sample = read_image(files[0])
    print(f"Found {len(files)} files under '{DATA_DIR}'. First file: {files[0]}")
    print(f"  shape={sample.shape}, range=[{sample.min():.3f}, {sample.max():.3f}]", flush=True)

    # Reproducible split on FILE indices; separate datasets so validation uses deterministic crops
    perm = torch.randperm(len(files), generator=torch.Generator().manual_seed(SEED)).tolist()
    n_train = int(0.85 * len(files))
    train_ds = Subset(S2NAIPDataset(files, PATCH_SIZE, SCALE, is_train=True), perm[:n_train])
    val_ds = Subset(S2NAIPDataset(files, PATCH_SIZE, SCALE, is_train=False), perm[n_train:])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | workers: {NUM_WORKERS} | "
          f"batch: {BATCH_SIZE} x accum {ACCUM_STEPS}", flush=True)

    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=USE_CUDA, persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)

    model = TerraLens(n_bands=4, dim=48, scale=SCALE, use_checkpointing=True).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} | AMP dtype: {AMP_DTYPE}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

    bicubic_psnr = bicubic_baseline_psnr(val_loader)
    print(f"Bicubic baseline Val PSNR: {bicubic_psnr:.2f} dB (the model must beat this)", flush=True)

    opt1 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best1 = run_stage(model, train_loader, val_loader, opt1, scaler, max_epochs=60, stage_name="Stage 1",
                      accum_steps=ACCUM_STEPS, patience=10,
                      save_path=os.path.join(CKPT_DIR, "terralens_stage1_best.pt"))

    # Stage 2 must beat Stage 1's best, otherwise we keep Stage 1's weights
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    best_psnr = run_stage(model, train_loader, val_loader, opt2, scaler, max_epochs=20, stage_name="Stage 2",
                          accum_steps=ACCUM_STEPS, patience=8, baseline=best1,
                          save_path=os.path.join(CKPT_DIR, "terralens_best.pt"))

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "terralens_best.pt"))
    print(f"\nFinal Best Validation PSNR Saved: {best_psnr:.2f} dB "
          f"(bicubic {bicubic_psnr:.2f} dB, gain {best_psnr - bicubic_psnr:+.2f} dB)")


if __name__ == "__main__":
    main()