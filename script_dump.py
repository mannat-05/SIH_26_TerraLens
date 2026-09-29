#!/usr/bin/env python3
"""
Terra Lens -- full dual-domain super-resolution training for Kaggle (2x T4, DDP)
Sentinel-2 (10 m, B02/B03/B04/B08) -> 2.5 m, trained on S2-NAIP RGB+NIR pairs.

Architecture follows the Terra Lens report (SIH.pdf):
  spatial branch  : shallow conv -> RSTB x N (edge-guided shifted-window attention)
  frequency branch: block-DCT -> FRN (conv) -> inverse block-DCT
  guided fusion   : gate g = sigmoid(conv(AC-energy map of the block-DCT))
  upsampler       : PixelShuffle (x4) -> conv, plus global bicubic residual

Launch (both T4s):   python -m torch.distributed.run --standalone --nproc_per_node=2 terralens_train.py [args]
Single GPU / CPU:    python terralens_train.py [args]
Inspect the dataset: python terralens_train.py --inspect_only --data_root /kaggle/input/...
Speed benchmark:     ... terralens_train.py --benchmark
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

BANDS = ["B02", "B03", "B04", "B08"]          # Blue, Green, Red, NIR (Sentinel-2 order used everywhere)
RANK = 0
LOGF = None
T_START = time.time()


def log(*a):
    if RANK == 0:
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        if LOGF is not None:
            LOGF.write(msg + "\n")
            LOGF.flush()


# =====================================================================================
# 0. Arguments
# =====================================================================================
PRESETS = {
    "tiny": dict(dim=16, heads=2, n_rstb=1, depth=2, up_feat=8, freq_feat=16),      # CPU smoke tests
    "base": dict(dim=64, heads=4, n_rstb=6, depth=2, up_feat=32, freq_feat=64),     # default (report: RSTB x6)
    "large": dict(dim=96, heads=6, n_rstb=8, depth=2, up_feat=48, freq_feat=96),
}


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- data / preprocessing
    p.add_argument("--data_root", default="")
    p.add_argument("--out", default="/kaggle/working/terralens_out")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--scale", type=int, default=4, help="SR factor. 4 => 10 m -> 2.5 m")
    p.add_argument("--hr_glob", default="", help="optional glob (relative to data_root) for HR files")
    p.add_argument("--lr_glob", default="", help="optional glob (relative to data_root) for LR files")
    p.add_argument("--hr_band_order", default="2,1,0,3", help="NAIP (R,G,B,NIR) -> (B,G,R,NIR)")
    p.add_argument("--lr_band_order", default="0,1,2,3")
    p.add_argument("--s2_mode", default="median", choices=["median", "first"],
                   help="how to collapse the stack of S2 revisits (median = cloud-robust composite)")
    p.add_argument("--max_tiles", type=int, default=0, help="0 = all")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--group_size", type=int, default=16, help="tiles per spatial block used for leak-free splits")
    p.add_argument("--io_threads", type=int, default=4)
    p.add_argument("--rebuild_cache", action="store_true")
    # ---- model
    p.add_argument("--preset", default="base", choices=list(PRESETS))
    for k in ["dim", "heads", "n_rstb", "depth", "up_feat", "freq_feat"]:
        p.add_argument(f"--{k}", type=int, default=None)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--fusion", default="feature", choices=["feature", "image"],
                   help="feature: fuse branch features then upsample (default, higher PSNR); "
                        "image: fuse 4-band images exactly as drawn in the report")
    # ---- optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--min_epochs", type=int, default=25)
    p.add_argument("--batch_per_gpu", type=int, default=16)
    p.add_argument("--patch", type=int, default=0, help="LR crop size (0 = whole tile); multiple of 8")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--no_amp", action="store_true")
    # ---- early stopping / schedule
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=0.005)
    p.add_argument("--lr_patience", type=int, default=7)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_hours", type=float, default=10.0, help="hard wall-clock budget (Kaggle limit is 12 h)")
    p.add_argument("--monitor", default="mean", choices=["real", "wald", "mean"])
    # ---- loss
    p.add_argument("--w_freq", type=float, default=0.10)
    p.add_argument("--w_sam", type=float, default=0.05)
    p.add_argument("--w_edge", type=float, default=0.05)
    p.add_argument("--w_cons", type=float, default=0.05)
    p.add_argument("--p_wald", type=float, default=0.30, help="fraction of training samples using a Wald-degraded LR")
    p.add_argument("--wald_sigma_lo", type=float, default=1.4)
    p.add_argument("--wald_sigma_hi", type=float, default=2.6)
    p.add_argument("--wald_sigma_eval", type=float, default=2.0)
    # ---- modes
    p.add_argument("--resume", action="store_true")
    p.add_argument("--inspect_only", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark_steps", type=int, default=30)
    p.add_argument("--assume_train_tiles", type=int, default=4000)
    p.add_argument("--zip_name", default="terralens_results.zip")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


# =====================================================================================
# 1. Raster IO (rasterio -> tifffile -> PIL -> numpy fallback chain)
# =====================================================================================
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz"}


def read_array(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        z = np.load(path)
        return z[z.files[0]]
    if ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        im = Image.open(path)
        if im.mode not in ("RGBA", "RGB", "L", "I;16"):
            im = im.convert("RGBA")
        return np.array(im)                                   # (H,W,C) -- RGBA keeps NIR in alpha
    # GeoTIFF
    if not os.environ.get("TERRALENS_NO_RASTERIO"):
        try:
            import rasterio
            with rasterio.open(path) as s:
                return s.read()                               # (C,H,W)
        except ImportError:
            pass
    import tifffile
    return tifffile.imread(str(path))


def to_chw(a):
    a = np.asarray(a)
    if a.ndim == 2:
        return a[None]
    if a.ndim != 3:
        raise ValueError(f"unsupported array rank {a.ndim} shape {a.shape}")
    s0, s1, s2 = a.shape
    if s1 == s2 and s0 != s1:
        return a                                              # already (C,H,W)
    if s0 == s1 and s2 != s1:
        return np.transpose(a, (2, 0, 1))                     # (H,W,C)
    if s0 == s1 == s2:
        return a
    # non-square tiles: channel axis is the smallest one
    ax = int(np.argmin(a.shape))
    return np.moveaxis(a, ax, 0)


def to_uint8(a):
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return np.round(a.astype(np.float32) / 257.0).astype(np.uint8)
    a = a.astype(np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


# =====================================================================================
# 2. Dataset discovery (pairs S2 <-> NAIP by tile id; never by list position)
# =====================================================================================
HR_TOK = {"naip", "hr", "high", "highres", "hires", "target", "gt", "label"}
LR_TOK = {"s2", "sentinel", "sentinel2", "sen2", "lr", "low", "lowres", "input"}
SPLIT_TOK = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
             "test": "test", "testing": "test"}
TILE_RE = re.compile(r"^(\d+_-?\d+_-?\d+)(?:_(8|16|32))?$")


def _toks(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def classify_role(path: Path, root: Path):
    rel = [p.lower() for p in path.relative_to(root).parts]
    for d in reversed(rel[:-1]):
        t = _toks(d)
        if any("naip" in x for x in t) or (t & HR_TOK):
            return "hr"
        if (t & LR_TOK) or any(x.startswith("sentinel") for x in t):
            return "lr"
    t = _toks(Path(rel[-1]).stem)
    if any("naip" in x for x in t) or (t & HR_TOK):
        return "hr"
    if (t & LR_TOK) or any(x.startswith("sentinel") for x in t):
        return "lr"
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        return "hr"
    if ext in (".tif", ".tiff"):
        return "lr"
    return None


def split_of(path: Path, root: Path):
    for part in path.relative_to(root).parts[:-1]:
        s = SPLIT_TOK.get(part.lower())
        if s:
            return s
    return "all"


def make_key(path: Path):
    """return (key, suffix). AllenAI ids look like 32612_960_-6049 (NAIP) / 32612_960_-6049_8 (S2 10 m bands)."""
    stem = path.stem
    m = TILE_RE.match(stem)
    if m:
        return m.group(1), m.group(2)
    s = stem.lower()
    for _ in range(3):
        s2 = re.sub(r"([_\-](hr|lr|naip|s2|sentinel2|sen2|rgbnir))$", "", s)
        s2 = re.sub(r"^((naip|s2|sentinel2|hr|lr)[_\-])", "", s2)
        if s2 == s:
            break
        s = s2
    return s, None


def tile_group(key, group_size):
    m = re.match(r"^(\d+)_(-?\d+)_(-?\d+)$", key)
    if m:
        utm, c, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{utm}_{c // group_size}_{r // group_size}"
    return key


def discover_pairs(root: Path, args):
    root = Path(root)
    if args.hr_glob or args.lr_glob:
        hr_files = sorted(root.glob(args.hr_glob)) if args.hr_glob else []
        lr_files = sorted(root.glob(args.lr_glob)) if args.lr_glob else []
        files = [(f, "hr") for f in hr_files] + [(f, "lr") for f in lr_files]
    else:
        files = []
        for dp, _, fn in os.walk(root):
            for f in fn:
                p = Path(dp) / f
                if p.suffix.lower() in IMG_EXT:
                    role = classify_role(p, root)
                    if role:
                        files.append((p, role))
    stats = defaultdict(int)
    cand = defaultdict(lambda: {"hr": [], "lr": []})
    for p, role in files:
        key, suf = make_key(p)
        sp = split_of(p, root)
        if role == "lr" and suf in ("16", "32"):
            stats["ignored_s2_20m_60m"] += 1
            continue
        cand[(sp, key)][role].append((p, suf))
        stats[f"files_{role}"] += 1
    pairs = []
    for (sp, key), d in sorted(cand.items()):
        if not d["hr"] or not d["lr"]:
            stats["unpaired"] += 1
            continue
        hr = sorted(d["hr"], key=lambda t: ("oldnaip" in str(t[0]).lower(), str(t[0])))[0][0]
        lr = sorted(d["lr"], key=lambda t: (t[1] != "8", str(t[0])))[0][0]
        pairs.append(dict(key=key, split=sp, hr=hr, lr=lr, group=tile_group(key, args.group_size)))
    return pairs, dict(stats), files


def print_inspection(root, pairs, stats, files):
    log("=" * 78)
    log(f"DATASET INSPECTION  root={root}")
    ext = defaultdict(int)
    for p, role in files:
        ext[(role, p.suffix.lower())] += 1
    for k, v in sorted(ext.items()):
        log(f"  files role={k[0]:2s} ext={k[1]:6s} : {v}")
    log(f"  stats: {stats}")
    sp = defaultdict(int)
    for p in pairs:
        sp[p["split"]] += 1
    log(f"  paired tiles: {len(pairs)}   per split: {dict(sp)}")
    for p in pairs[:3]:
        log(f"    e.g. key={p['key']}\n         HR={p['hr']}\n         LR={p['lr']}")
    log("=" * 78)


def tree_dump(root, n=40):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in sorted(fn):
            out.append(str(Path(dp, f).relative_to(root)))
            if len(out) >= n:
                return out
    return out


# =====================================================================================
# 3. Preprocessing -> on-disk cache (uint8 HR at 4x, float16 S2 composite at 1x)
# =====================================================================================
def load_lr(path, band_perm, mode):
    a = to_chw(read_array(path))
    C, H, W = a.shape
    if C % 4 != 0:
        raise ValueError(f"S2 file has {C} channels, expected a multiple of 4 (bands x revisits): {path}")
    T = C // 4
    a = a.reshape(T, 4, H, W).astype(np.float32)
    x = np.median(a, axis=0) if (mode == "median" and T > 1) else a[0]
    x = x[list(band_perm)]
    x = x / (10000.0 if x.max() > 5.0 else 1.0)              # L1C DN -> reflectance (fixed rule per tile)
    return np.clip(np.nan_to_num(x), 0.0, 1.0).astype(np.float16), T


def load_hr(path, band_perm, out_hw):
    a = to_chw(read_array(path))
    if a.shape[0] < 4:
        raise ValueError(f"HR file has {a.shape[0]} channels (need RGB+NIR): {path}")
    a = to_uint8(a[:4])[list(band_perm)]
    if a.shape[-2:] != tuple(out_hw):
        t = torch.from_numpy(a).float().unsqueeze(0)
        t = F.interpolate(t, size=tuple(out_hw), mode="area")   # area = alias-free anti-aliased downsample
        a = t[0].round().clamp(0, 255).byte().numpy()
    return a


def assign_splits(pairs, val_frac, test_frac, seed):
    """Use folder splits when present, otherwise carve val/test out of the pool by *spatial group*."""
    have = {p["split"] for p in pairs}
    pool = [p for p in pairs if p["split"] in ("train", "all")]
    fixed = {s: [p for p in pairs if p["split"] == s] for s in ("val", "test")}
    groups = sorted({p["group"] for p in pool}, key=lambda g: hashlib.md5(f"{seed}{g}".encode()).hexdigest())
    n = len(pool)
    need = {"val": ("val" not in have), "test": ("test" not in have)}
    quota = {"val": int(round(n * val_frac)) if need["val"] else 0,
             "test": int(round(n * test_frac)) if need["test"] else 0}
    by_g = defaultdict(list)
    for p in pool:
        by_g[p["group"]].append(p)
    out = {"train": [], "val": list(fixed["val"]), "test": list(fixed["test"])}
    cnt = {"val": 0, "test": 0}
    for g in groups:
        tgt = "train"
        for s in ("test", "val"):
            if need[s] and cnt[s] < quota[s]:
                tgt = s
                break
        out[tgt].extend(by_g[g])
        if tgt != "train":
            cnt[tgt] += len(by_g[g])
    for k in out:
        out[k].sort(key=lambda p: p["key"])
    return out


def cache_id(pairs, args):
    h = hashlib.md5()
    h.update(json.dumps([args.scale, args.hr_band_order, args.lr_band_order, args.s2_mode, args.max_tiles,
                         args.val_frac, args.test_frac, args.group_size, args.seed, len(pairs)]).encode())
    for p in pairs[:25] + pairs[-25:]:
        h.update(p["key"].encode())
    return h.hexdigest()[:12]


def build_cache(args, cache_dir: Path, root: Path):
    pairs, st, files = discover_pairs(root, args)
    print_inspection(root, pairs, st, files)
    if not pairs:
        raise RuntimeError("No (HR, LR) pairs found under %s.\nFirst files seen:\n  %s\n"
                           "Fix: pass --hr_glob / --lr_glob (paths relative to --data_root), e.g. "
                           "--hr_glob '**/naip/*.png' --lr_glob '**/sentinel2/*_8.tif'"
                           % (root, "\n  ".join(tree_dump(root))))
    if args.max_tiles:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = sorted(pairs[: args.max_tiles], key=lambda p: p["key"])
    cid = cache_id(pairs, args)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not args.rebuild_cache:
        try:
            if json.load(open(meta_path)).get("cache_id") == cid:
                log(f"[cache] reusing {cache_dir} (id {cid})")
                return
        except Exception:
            pass
    splits = assign_splits(pairs, args.val_frac, args.test_frac, args.seed)
    ordered = splits["train"] + splits["val"] + splits["test"]
    if not splits["train"]:
        raise RuntimeError("Split produced 0 training tiles; lower --val_frac/--test_frac or check the dataset.")
    hp = [int(x) for x in args.hr_band_order.split(",")]
    lp = [int(x) for x in args.lr_band_order.split(",")]
    # first readable tile fixes the geometry
    lr0, T0 = load_lr(ordered[0]["lr"], lp, args.s2_mode)
    Hl, Wl = lr0.shape[-2:]
    if Hl % args.window != 0 or Wl % args.window != 0:
        raise RuntimeError(f"LR tile {Hl}x{Wl} is not a multiple of the attention window {args.window}")
    Hh, Wh = Hl * args.scale, Wl * args.scale
    N = len(ordered)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lr_mm = np.lib.format.open_memmap(cache_dir / "lr.npy", mode="w+", dtype=np.float16, shape=(N, 4, Hl, Wl))
    hr_mm = np.lib.format.open_memmap(cache_dir / "hr.npy", mode="w+", dtype=np.uint8, shape=(N, 4, Hh, Wh))
    ok = np.zeros(N, dtype=bool)
    revisits = np.zeros(N, dtype=np.int32)
    errs = []

    def work(i):
        p = ordered[i]
        try:
            lr, T = load_lr(p["lr"], lp, args.s2_mode)
            if lr.shape[-2:] != (Hl, Wl):
                raise ValueError(f"LR size {lr.shape[-2:]} != {(Hl, Wl)}")
            hr = load_hr(p["hr"], hp, (Hh, Wh))
            lr_mm[i] = lr
            hr_mm[i] = hr
            ok[i] = True
            revisits[i] = T
        except Exception as e:                                # noqa
            errs.append((p["key"], repr(e)))

    t0 = time.time()
    log(f"[cache] preprocessing {N} tiles -> {cache_dir}  (LR {Hl}x{Wl} float16 composite of <= {T0} revisits, "
        f"HR {Hh}x{Wh} uint8)")
    with ThreadPoolExecutor(max_workers=args.io_threads) as ex:
        for k, _ in enumerate(ex.map(work, range(N))):
            if (k + 1) % 500 == 0:
                log(f"   {k + 1}/{N} tiles  ({time.time() - t0:.0f}s)")
    lr_mm.flush()
    hr_mm.flush()
    n_ok = int(ok.sum())
    log(f"[cache] done in {time.time() - t0:.0f}s: {n_ok}/{N} tiles usable, {len(errs)} failed")
    if errs:
        log("   first failures:", errs[:3])
    if n_ok < 8:
        raise RuntimeError(f"Only {n_ok} tiles could be read. First errors: {errs[:3]}")
    counts = {s: 0 for s in ("train", "val", "test")}
    pos = 0
    keep_idx = []
    ranges = {}
    for s in ("train", "val", "test"):
        n_s = len(splits[s])
        idxs = [i for i in range(pos, pos + n_s) if ok[i]]
        keep_idx.extend(idxs)
        ranges[s] = [len(keep_idx) - len(idxs), len(keep_idx)]
        counts[s] = len(idxs)
        pos += n_s
    if len(keep_idx) != N:                                    # compact the cache
        keep = np.array(keep_idx)
        lr_c = np.ascontiguousarray(lr_mm[keep])
        hr_c = np.ascontiguousarray(hr_mm[keep])
        del lr_mm, hr_mm
        np.save(cache_dir / "lr.npy", lr_c)
        np.save(cache_dir / "hr.npy", hr_c)
        lr_mm, hr_mm = lr_c, hr_c
    keys = [ordered[i]["key"] for i in keep_idx]
    n_tr = ranges["train"][1]
    # dataset statistics (train split only) used for radiometric harmonisation
    sub = np.linspace(0, n_tr - 1, num=min(n_tr, 1500)).astype(int)
    lr_s = np.asarray(lr_mm[sub]).astype(np.float32)
    hr_s = np.asarray(hr_mm[sub]).astype(np.float32) / 255.0
    stats = dict(
        s2_mean=lr_s.mean((0, 2, 3)).tolist(), s2_std=lr_s.std((0, 2, 3)).tolist(),
        naip_mean=hr_s.mean((0, 2, 3)).tolist(), naip_std=hr_s.std((0, 2, 3)).tolist())
    diag = alignment_diagnostics(lr_s[:200], hr_s[:200], args.scale)
    meta = dict(cache_id=cid, n=len(keep_idx), ranges=ranges, counts=counts, keys=keys, stats=stats,
                lr_hw=[Hl, Wl], hr_hw=[Hh, Wh], revisits_median=float(np.median(revisits[ok])),
                revisits_min=int(revisits[ok].min()), revisits_max=int(revisits[ok].max()),
                diagnostics=diag, failed=len(errs))
    json.dump(meta, open(meta_path, "w"), indent=1)
    log(f"[cache] split sizes: {counts} | S2 revisits/tile: median {meta['revisits_median']:.0f} "
        f"(min {meta['revisits_min']}, max {meta['revisits_max']})")
    log(f"[cache] S2  mean/std per band {np.round(stats['s2_mean'], 4).tolist()} / {np.round(stats['s2_std'], 4).tolist()}")
    log(f"[cache] NAIP mean/std per band {np.round(stats['naip_mean'], 4).tolist()} / {np.round(stats['naip_std'], 4).tolist()}")
    for k, v in diag.items():
        log(f"[diag] {k}: {v}")


def alignment_diagnostics(lr, hr, scale):
    """Sanity checks that catch the classic silent data bugs: band order, orientation, sub-tile shifts."""
    hr_lo = F.avg_pool2d(torch.from_numpy(hr), scale).numpy()                 # HR at LR grid
    out = {}
    B, C = lr.shape[0], 4
    corr = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            a = lr[:, i].reshape(B, -1)
            b = hr_lo[:, j].reshape(B, -1)
            a = a - a.mean(1, keepdims=True)
            b = b - b.mean(1, keepdims=True)
            c = (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-8)
            corr[i, j] = np.nanmean(c)
    out["band_corr_diag(LR_i vs HR_i)"] = np.round(np.diag(corr), 3).tolist()
    best = corr.argmax(1).tolist()
    out["best_HR_band_for_each_LR_band"] = best
    if best != [0, 1, 2, 3]:
        out["WARNING_band_order"] = ("HR band order may not match S2 (B,G,R,NIR). Check --hr_band_order. "
                                     "Blue/green are highly correlated so this is only a hint.")
    g_lr = torch.from_numpy(lr[:, :3].mean(1))
    g_hr = torch.from_numpy(hr_lo[:, :3].mean(1))
    def zc(x):
        x = x - x.mean((1, 2), keepdim=True)
        return x / (x.std((1, 2), keepdim=True) + 1e-8)
    a = zc(g_lr)
    res = {}
    for name, fn in {"identity": lambda x: x, "flipH": lambda x: x.flip(-1), "flipV": lambda x: x.flip(-2),
                     "transpose": lambda x: x.transpose(-1, -2), "rot90": lambda x: torch.rot90(x, 1, (-2, -1)),
                     "rot180": lambda x: torch.rot90(x, 2, (-2, -1)), "rot270": lambda x: torch.rot90(x, 3, (-2, -1))}.items():
        res[name] = float((a * zc(fn(g_hr))).mean())
    out["orientation_corr"] = {k: round(v, 3) for k, v in res.items()}
    if max(res, key=res.get) != "identity" and res["identity"] < res[max(res, key=res.get)] - 0.05:
        out["WARNING_orientation"] = f"HR looks {max(res, key=res.get)}-transformed relative to LR"
    shifts = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sh = torch.roll(g_hr, (dy, dx), (1, 2))[:, 3:-3, 3:-3]
            shifts[(dy, dx)] = float((a[:, 3:-3, 3:-3] * zc(sh)).mean())
    bs = max(shifts, key=shifts.get)
    out["best_LR_pixel_shift(dy,dx)"] = bs
    out["corr_at_zero_shift"] = round(shifts[(0, 0)], 3)
    if bs != (0, 0):
        out["WARNING_shift"] = f"HR appears shifted by {bs} LR pixels relative to S2"
    return out


class TileStore:
    def __init__(self, cache_dir, mmap=True):
        cache_dir = Path(cache_dir)
        self.meta = json.load(open(cache_dir / "meta.json"))
        mode = "r" if mmap else None
        self.lr = np.load(cache_dir / "lr.npy", mmap_mode=mode)
        self.hr = np.load(cache_dir / "hr.npy", mmap_mode=mode)
        self.ranges = self.meta["ranges"]
        self.stats = self.meta["stats"]
        self.keys = self.meta["keys"]

    def n(self, split):
        a, b = self.ranges[split]
        return b - a

    def get(self, idx, device, harm):
        idx = np.sort(np.asarray(idx))
        lr = torch.from_numpy(np.ascontiguousarray(self.lr[idx])).to(device, non_blocking=True).float()
        hr = torch.from_numpy(np.ascontiguousarray(self.hr[idx])).to(device, non_blocking=True).float().div_(255.0)
        return lr, harm(hr)


class Harmonizer:
    """Moment-match NAIP (8-bit DN) to Sentinel-2 reflectance statistics, per band.  Keeps SR output in S2 radiometry."""
    def __init__(self, stats, device):
        f = lambda k: torch.tensor(stats[k], dtype=torch.float32, device=device).view(1, 4, 1, 1)
        self.sm, self.ss, self.nm, self.ns = f("s2_mean"), f("s2_std"), f("naip_mean"), f("naip_std")

    def __call__(self, hr):
        return ((hr - self.nm) / (self.ns + 1e-6) * self.ss + self.sm).clamp_(0.0, 1.0)


# =====================================================================================
# 4. Model (Terra Lens, per report)
# =====================================================================================
class BlockDCT(nn.Module):
    """Orthonormal 8x8 block DCT-II, coefficients kept in block layout (same tensor shape as input)."""
    def __init__(self, block=8):
        super().__init__()
        self.b = block
        D = torch.zeros(block, block)
        for k in range(block):
            ck = math.sqrt(1.0 / block) if k == 0 else math.sqrt(2.0 / block)
            for i in range(block):
                D[k, i] = ck * math.cos(math.pi * (2 * i + 1) * k / (2 * block))
        self.register_buffer("D", D, persistent=False)

    def forward(self, x, inverse=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, C, H, W = x.shape
            b = self.b
            ph, pw = (-H) % b, (-W) % b
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            Hp, Wp = x.shape[-2:]
            M = self.D.t() if inverse else self.D
            x6 = x.reshape(B, C, Hp // b, b, Wp // b, b)
            t = torch.einsum("ui,bcpiqj->bcpuqj", M, x6)
            y = torch.einsum("vj,bcpuqj->bcpuqv", M, t)
            return y.reshape(B, C, Hp, Wp)[:, :, :H, :W]


def ac_energy_map(D, b=8):
    """Mean AC (non-DC) DCT energy per 8x8 block, relative-normalised, smoothly upsampled to full size."""
    with torch.autocast(device_type=D.device.type, enabled=False):
        e = (D.float() ** 2).mean(1, keepdim=True)
        H, W = e.shape[-2:]
        mask = torch.ones(b, b, device=e.device)
        mask[0, 0] = 0
        mask = mask.repeat(H // b, W // b)[None, None]
        blk = F.avg_pool2d(e * mask, b) * (b * b) / (b * b - 1)
        blk = torch.log1p(blk / (blk.mean((2, 3), keepdim=True) + 1e-8))
        return F.interpolate(blk, size=(H, W), mode="bilinear", align_corners=False)


class SobelEdge(nn.Module):
    """Edge map E = sigmoid(k * (|grad| / mean|grad| - 1)).  The per-image normalisation matters: raw reflectance
    gradients are ~0.01-0.05, so sigmoid(|grad|) alone is ~0.5 everywhere (i.e. no guidance at all)."""
    def __init__(self, n_bands=4):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1), persistent=False)
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3).repeat(n_bands, 1, 1, 1), persistent=False)
        self.n = n_bands

    @torch.no_grad()
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.pad(x.float(), (1, 1, 1, 1), mode="reflect")
            gx = F.conv2d(x, self.kx, groups=self.n)
            gy = F.conv2d(x, self.ky, groups=self.n)
            mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).mean(1, keepdim=True)
            rel = mag / (mag.mean((2, 3), keepdim=True) + 1e-6)
            return torch.sigmoid(2.0 * (rel - 1.0))


def sobel_mag(x):
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype)
    C = x.shape[1]
    gx = F.conv2d(xp, kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1), groups=C)
    gy = F.conv2d(xp, kx.t().contiguous().view(1, 1, 3, 3).repeat(C, 1, 1, 1), groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def win_part(x, w):
    B, C, H, W = x.shape
    return x.view(B, C, H // w, w, W // w, w).permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, C)


def win_rev(t, w, B, C, H, W):
    return t.view(B, H // w, W // w, w, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)


_MASKS = {}


def shift_mask(Hp, Wp, w, s, device):
    key = (Hp, Wp, w, s, str(device))
    if key not in _MASKS:
        img = torch.zeros(1, 1, Hp, Wp)
        cnt = 0
        for hs in (slice(0, -w), slice(-w, -s), slice(-s, None)):
            for ws_ in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                img[:, :, hs, ws_] = cnt
                cnt += 1
        mw = win_part(img, w).squeeze(-1)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        am = am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)
        _MASKS[key] = am.to(device)
    return _MASKS[key]


class EdgeWindowAttention(nn.Module):
    """Swin window attention with an edge prior:  softmax(QK^T/sqrt(d)+B) * (1 + alpha_h * E), re-normalised."""
    def __init__(self, dim, window, heads):
        super().__init__()
        self.h, self.w = heads, window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0).contiguous()
        r[:, :, 0] += window - 1
        r[:, :, 1] += window - 1
        r[:, :, 0] *= 2 * window - 1
        self.register_buffer("rel_index", r.sum(-1), persistent=False)
        self.alpha_raw = nn.Parameter(torch.full((heads,), math.log(math.e ** 0.5 - 1.0)))   # softplus -> 0.5

    def forward(self, x, e, B, mask=None):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel[self.rel_index.view(-1)].view(N, N, self.h).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B, nW, self.h, N, N) + mask.unsqueeze(1).unsqueeze(0)).view(Bn, self.h, N, N)
        attn = attn.softmax(-1)
        gate = 1.0 + F.softplus(self.alpha_raw).view(1, self.h, 1, 1) * e.transpose(1, 2).unsqueeze(1)
        attn = attn * gate
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=2.0):
        super().__init__()
        self.w, self.s = window, shift
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = EdgeWindowAttention(dim, window, heads)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x, e):
        B, C, H, W = x.shape
        w, s = self.w, self.s
        ph, pw = (-H) % w, (-W) % w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            e = F.pad(e, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = x.shape[-2:]
        mask = None
        if s:
            x = torch.roll(x, (-s, -s), (2, 3))
            e = torch.roll(e, (-s, -s), (2, 3))
            mask = shift_mask(Hp, Wp, w, s, x.device)
        t, et = win_part(x, w), win_part(e, w)
        t = t + self.attn(self.n1(t), et, B, mask)
        t = t + self.mlp(self.n2(t))
        x = win_rev(t, w, B, C, Hp, Wp)
        if s:
            x = torch.roll(x, (s, s), (2, 3))
        return x[:, :, :H, :W]


class RSTB(nn.Module):
    def __init__(self, dim, depth, window, heads):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, window, 0 if i % 2 == 0 else window // 2)
                                     for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, e):
        r = x
        for b in self.blocks:
            x = b(x, e)
        return self.conv(x) + r


class TerraLens(nn.Module):
    def __init__(self, n_bands=4, dim=64, heads=4, n_rstb=6, depth=2, window=8, scale=4,
                 up_feat=32, freq_feat=64, fusion="feature", **_):
        super().__init__()
        assert scale in (2, 4, 8), "PixelShuffle upsampler supports x2/x4/x8"
        self.scale, self.fusion, self.n_bands = scale, fusion, n_bands
        self.edge = SobelEdge(n_bands)
        self.dct = BlockDCT(8)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # ---- spatial branch
        self.shallow = nn.Conv2d(n_bands, dim, 3, 1, 1)
        self.rstbs = nn.ModuleList([RSTB(dim, depth, window, heads) for _ in range(n_rstb)])
        self.conv_after = nn.Conv2d(dim, dim, 3, 1, 1)
        # ---- frequency branch (FRN in the block-DCT domain)
        self.freq_in = nn.Conv2d(n_bands, freq_feat, 3, 1, 1)
        self.freq_mid = nn.Conv2d(freq_feat, freq_feat, 3, 1, 1)
        fo = dim if fusion == "feature" else n_bands
        self.freq_out = nn.Conv2d(freq_feat, fo, 3, 1, 1)
        # ---- guided fusion gate from the AC-energy map
        self.gate = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(16, 1, 1))
        # ---- upsampler
        if fusion == "feature":
            self.up_pre = nn.Conv2d(dim, up_feat, 3, 1, 1)
            stages = []
            for _ in range(int(math.log2(scale))):
                stages += [nn.Conv2d(up_feat, up_feat * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, inplace=True)]
            self.up = nn.Sequential(*stages)
        else:
            self.spatial_out = nn.Conv2d(dim, n_bands, 3, 1, 1)
            self.up = nn.Sequential(nn.Conv2d(n_bands, up_feat * scale * scale, 3, 1, 1),
                                    nn.PixelShuffle(scale), nn.LeakyReLU(0.1, inplace=True))
        self.conv_last = nn.Conv2d(up_feat, n_bands, 3, 1, 1)

    def forward(self, lr, return_aux=False):
        e = self.edge(lr)
        D = self.dct(lr)
        g = torch.sigmoid(self.gate(ac_energy_map(D)))
        s0 = self.shallow(lr)
        h = s0
        for r in self.rstbs:
            h = r(h, e)
        Fs = self.conv_after(h) + s0
        f = self.act(self.freq_in(D))
        f = self.act(self.freq_mid(f)) + f
        Ff = self.dct(self.freq_out(f), inverse=True)
        if self.fusion == "feature":
            fused = g * Ff + (1.0 - g) * Fs
            x = self.up(self.act(self.up_pre(fused)))
        else:
            fused = g * Ff + (1.0 - g) * self.spatial_out(Fs)
            x = self.up(fused)
        sr = self.conv_last(x) + F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return (sr, {"gate": g, "edge": e}) if return_aux else sr


def build_model(args):
    return TerraLens(n_bands=4, dim=args.dim, heads=args.heads, n_rstb=args.n_rstb, depth=args.depth,
                     window=args.window, scale=args.scale, up_feat=args.up_feat, freq_feat=args.freq_feat,
                     fusion=args.fusion)


def model_cfg(args):
    return dict(dim=args.dim, heads=args.heads, n_rstb=args.n_rstb, depth=args.depth, window=args.window,
                scale=args.scale, up_feat=args.up_feat, freq_feat=args.freq_feat, fusion=args.fusion)


# =====================================================================================
# 5. Degradation (Wald), losses, metrics
# =====================================================================================
class Degrade:
    """Gaussian blur + decimation, half-pixel aligned (the LR sample sits at the centre of its s x s footprint)."""
    def __init__(self, scale):
        self.s = scale

    def __call__(self, x, sigma=2.0):
        s, C = self.s, x.shape[1]
        K = 2 * s
        ax = torch.arange(K, device=x.device, dtype=x.dtype) - (K - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).expand(C, 1, K, K).contiguous()
        xp = F.pad(x, (s // 2,) * 4, mode="reflect")
        return F.conv2d(xp, k, stride=s, groups=C)


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def sam_rad(a, b, eps=1e-6):
    cos = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


class Criterion:
    def __init__(self, args, dct, degrade):
        self.a, self.dct, self.deg = args, dct, degrade

    def __call__(self, sr, hr, lr):
        a = self.a
        l_pix = charbonnier(sr, hr)
        l_frq = charbonnier(self.dct(sr), self.dct(hr))
        l_sam = sam_rad(sr, hr)
        l_edg = F.l1_loss(sobel_mag(sr), sobel_mag(hr))
        l_con = F.l1_loss(self.deg(sr, a.wald_sigma_eval), lr)
        total = l_pix + a.w_freq * l_frq + a.w_sam * l_sam + a.w_edge * l_edg + a.w_cons * l_con
        return total, dict(pix=l_pix.item(), freq=l_frq.item(), sam=l_sam.item(), edge=l_edg.item(), cons=l_con.item())


def _gauss_win(C, device, size=11, sigma=1.5):
    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(C, 1, size, size).contiguous()


def ssim_img(x, y):
    C = x.shape[1]
    w = _gauss_win(C, x.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = F.conv2d(x, w, padding=5, groups=C), F.conv2d(y, w, padding=5, groups=C)
    sxx = F.conv2d(x * x, w, padding=5, groups=C) - mx ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=C) - my ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=C) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return m.mean((1, 2, 3))


def metrics(sr, hr, scale):
    """per-image metric vectors, all in float32 on [0,1] reflectance."""
    mse = ((sr - hr) ** 2).mean((1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    sam = torch.acos(((sr * hr).sum(1) / (sr.norm(dim=1) * hr.norm(dim=1) + 1e-6)).clamp(-1 + 1e-6, 1 - 1e-6)).mean((1, 2)) * 180 / math.pi
    rmse_b = ((sr - hr) ** 2).mean((2, 3)).sqrt()
    mu_b = hr.mean((2, 3)) + 1e-6
    ergas = 100.0 / scale * ((rmse_b / mu_b) ** 2).mean(1).sqrt()
    nd = lambda x: (x[:, 3] - x[:, 2]) / (x[:, 3] + x[:, 2] + 1e-6)
    ndvi = (nd(sr) - nd(hr)).abs().mean((1, 2))
    return dict(psnr=psnr, ssim=ssim_img(sr, hr), sam=sam, ergas=ergas, ndvi_mae=ndvi)


@torch.no_grad()
def evaluate(model, store, split, harm, args, device, amp, batch=32, max_tiles=0):
    model.eval()
    deg = Degrade(args.scale)
    a, b = store.ranges[split]
    if max_tiles:
        b = min(b, a + max_tiles)
    acc, n = defaultdict(float), 0
    for s in range(a, b, batch):
        idx = np.arange(s, min(s + batch, b))
        lr, hr = store.get(idx, device, harm)
        B = lr.shape[0]
        legs = {"real": (lr, hr), "wald": (deg(hr, args.wald_sigma_eval), hr)}
        for leg, (x, y) in legs.items():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                sr = model(x)
            sr = sr.float().clamp(0, 1)
            bic = F.interpolate(x, scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0, 1)
            m, mb = metrics(sr, y, args.scale), metrics(bic, y, args.scale)
            for k in m:
                acc[f"{leg}_{k}"] += m[k].sum().item()
                acc[f"{leg}_bicubic_{k}"] += mb[k].sum().item()
            if leg == "real":
                acc["consistency_L1"] += F.l1_loss(deg(sr, args.wald_sigma_eval), x, reduction="none").mean((1, 2, 3)).sum().item()
        n += B
    return {k: v / max(n, 1) for k, v in acc.items()}


# =====================================================================================
# 6. Training utilities
# =====================================================================================
class EMA:
    def __init__(self, model, decay):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        ep, mp = list(self.m.parameters()), [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ep, d)
        torch._foreach_add_(ep, mp, alpha=1 - d)
        for be, bm in zip(self.m.buffers(), model.buffers()):
            be.copy_(bm)


def dihedral(lr, hr):
    B = lr.shape[0]
    ks = torch.randint(0, 8, (B,), device=lr.device)
    o_lr, o_hr = torch.empty_like(lr), torch.empty_like(hr)
    for k in range(8):
        m = ks == k
        if not bool(m.any()):
            continue
        a, b = lr[m], hr[m]
        if k & 4:
            a, b = a.flip(-1), b.flip(-1)
        r = k & 3
        if r:
            a, b = torch.rot90(a, r, (-2, -1)), torch.rot90(b, r, (-2, -1))
        o_lr[m], o_hr[m] = a, b
    return o_lr, o_hr


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_flops(model, hw, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.rand(1, 4, hw[0], hw[1], device=device)
        model.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x)
        return fc.get_total_flops()
    except Exception:
        return None


def setup_dist():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    lrank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda", lrank if ws > 1 else 0) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(dev)
        torch.backends.cudnn.benchmark = True
    if ws > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=90))
    return rank, ws, lrank, dev


def find_data_root(args):
    if args.data_root:
        return Path(args.data_root)
    base = Path("/kaggle/input")
    if base.exists():
        hits = [p for p in base.rglob("*") if p.is_dir() and re.search(r"s2.?naip", p.name, re.I)]
        if hits:
            return sorted(hits, key=lambda p: len(p.parts))[0]
        subs = [p for p in base.iterdir() if p.is_dir()]
        if len(subs) == 1:
            return subs[0]
    raise FileNotFoundError("Could not locate the dataset; pass --data_root")


def default_cache_dir(args):
    if args.cache_dir:
        return Path(args.cache_dir)
    for base in ("/kaggle/temp", "/tmp"):
        if os.path.isdir(base):
            return Path(base) / "terralens_cache"
    return Path(args.out) / "cache"


def wait_for(path, timeout=7200):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(5)


# =====================================================================================
# 7. Benchmark (no dataset needed)
# =====================================================================================
def run_benchmark(args, rank, ws, lrank, device):
    amp = device.type == "cuda" and not args.no_amp
    model = build_model(args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    fl = count_flops(model, (64, 64), device)
    log(f"[bench] params {n_par / 1e6:.2f}M | forward {fl / 1e9:.1f} GFLOPs per 64x64 tile" if fl else f"[bench] params {n_par / 1e6:.2f}M")
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[lrank] if device.type == "cuda" else None) if ws > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = make_scaler(amp)
    deg = Degrade(args.scale)
    crit = Criterion(args, BlockDCT(8).to(device), deg)
    B, H = args.batch_per_gpu, 64
    lr = torch.rand(B, 4, H, H, device=device) * 0.3
    hr = torch.rand(B, 4, H * args.scale, H * args.scale, device=device) * 0.3
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            sr = ddp(lr)
        loss, _ = crit(sr.float(), hr, lr)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.benchmark_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / args.benchmark_steps
    sps = B * ws / dt
    log(f"[bench] world={ws} batch/gpu={B}: {dt * 1000:.0f} ms/step -> {sps:.0f} tiles/s (global)")
    if device.type == "cuda":
        log(f"[bench] peak VRAM per GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    ep = args.assume_train_tiles / sps
    log(f"[bench] projected: {ep:.0f} s/epoch for {args.assume_train_tiles} train tiles "
        f"(+ eval) -> 100 epochs ~ {100 * ep / 3600:.2f} h | 200 epochs ~ {200 * ep / 3600:.2f} h | 300 epochs ~ {300 * ep / 3600:.2f} h")


# =====================================================================================
# 8. Training
# =====================================================================================
def train(args, store, rank, ws, lrank, device):
    out = Path(args.out)
    amp = device.type == "cuda" and not args.no_amp
    harm = Harmonizer(store.stats, device)
    model = build_model(args).to(device)
    ema = EMA(model, args.ema)
    n_par = sum(p.numel() for p in model.parameters())
    fl = count_flops(copy.deepcopy(model), tuple(store.meta["lr_hw"]), device)
    log(f"[model] Terra Lens {args.preset} fusion={args.fusion}: {n_par / 1e6:.2f}M params" +
        (f", {fl / 1e9:.1f} GFLOPs / tile forward" if fl else ""))
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[lrank] if device.type == "cuda" else None) if ws > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.lr_factor,
                                                          patience=args.lr_patience, min_lr=args.min_lr)
    scaler = make_scaler(amp)
    deg = Degrade(args.scale)
    crit = Criterion(args, BlockDCT(8).to(device), deg)

    hist, best, bad, start_ep, gstep = [], -1e9, 0, 1, 0
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        ema.m.load_state_dict(ck["ema"])
        ema.n = ck.get("ema_n", 0)
        opt.load_state_dict(ck["opt"])
        plateau.load_state_dict(ck["plateau"])
        scaler.load_state_dict(ck["scaler"])
        hist, best, bad, start_ep, gstep = ck["hist"], ck["best"], ck["bad"], ck["epoch"] + 1, ck["gstep"]
        log(f"[resume] continuing from epoch {start_ep} (best {best:.3f}, patience used {bad})")

    n_tr = store.n("train")
    bpg = args.batch_per_gpu
    gbs = bpg * ws
    steps = n_tr // gbs
    if steps < 1:
        raise RuntimeError(f"{n_tr} train tiles < global batch {gbs}; lower --batch_per_gpu")
    log(f"[train] {n_tr} train / {store.n('val')} val / {store.n('test')} test tiles | global batch {gbs} | "
        f"{steps} steps/epoch | AMP={amp} | budget {args.max_hours} h")
    base_lr = args.lr
    stop_flag = torch.zeros(2, device=device)                  # [stop, improved]
    Hl = store.meta["lr_hw"][0]

    for epoch in range(start_ep, args.epochs + 1):
        t0 = time.time()
        ddp.train()
        g = np.random.default_rng(args.seed * 1000 + epoch)
        perm = g.permutation(n_tr)[: steps * gbs].reshape(steps, ws, bpg)[:, rank, :]
        run = defaultdict(float)
        for si in range(steps):
            lr_b, hr_b = store.get(perm[si] + store.ranges["train"][0], device, harm)
            if args.patch and args.patch < Hl:
                p = args.patch
                y = 8 * random.randint(0, (Hl - p) // 8)
                x = 8 * random.randint(0, (Hl - p) // 8)
                lr_b = lr_b[..., y:y + p, x:x + p]
                hr_b = hr_b[..., y * args.scale:(y + p) * args.scale, x * args.scale:(x + p) * args.scale]
            lr_b, hr_b = dihedral(lr_b, hr_b)
            if args.p_wald > 0:
                m = torch.rand(lr_b.shape[0], device=device) < args.p_wald
                if bool(m.any()):
                    sig = random.uniform(args.wald_sigma_lo, args.wald_sigma_hi)
                    syn = deg(hr_b[m], sig)
                    syn = syn + torch.randn_like(syn) * random.uniform(0.0, 0.003)
                    lr_b = lr_b.clone()
                    lr_b[m] = syn.clamp(0, 1)
            gstep += 1
            if gstep <= args.warmup_steps:
                for gr in opt.param_groups:
                    gr["lr"] = base_lr * gstep / args.warmup_steps
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                sr = ddp(lr_b)
            loss, parts = crit(sr.float(), hr_b, lr_b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run["loss"] += loss.item()
            for k, v in parts.items():
                run[k] += v
        tr_time = time.time() - t0
        rec = {"epoch": epoch, **{k: v / steps for k, v in run.items()}, "lr": opt.param_groups[0]["lr"],
               "train_seconds": tr_time}
        stop = False
        improved = False
        if rank == 0:
            ev = evaluate(ema.m, store, "val", harm, args, device, amp)
            score = {"real": ev["real_psnr"], "wald": ev["wald_psnr"],
                     "mean": 0.5 * (ev["real_psnr"] + ev["wald_psnr"])}[args.monitor]
            rec.update({f"val_{k}": v for k, v in ev.items()})
            rec["monitor"] = score
            if gstep > args.warmup_steps:
                plateau.step(score)
            improved = score > best + args.min_delta
            if improved:
                best, bad = score, 0
                atomic_save({"ema": ema.m.state_dict(), "cfg": model_cfg(args), "stats": store.stats,
                             "epoch": epoch, "val": ev, "args": vars(args)}, out / "best.pt")
            else:
                bad += 1
            hours = (time.time() - T_START) / 3600
            stop = (bad >= args.patience and epoch >= args.min_epochs) or hours >= args.max_hours
            rec.update(best=best, bad=bad, hours=hours)
            hist.append(rec)
            atomic_save({"model": model.state_dict(), "ema": ema.m.state_dict(), "ema_n": ema.n, "opt": opt.state_dict(),
                         "plateau": plateau.state_dict(), "scaler": scaler.state_dict(), "hist": hist, "best": best,
                         "bad": bad, "epoch": epoch, "gstep": gstep}, last)
            write_history(out, hist)
            log(f"[E{epoch:03d}] loss {rec['loss']:.4f} (pix {rec['pix']:.4f} frq {rec['freq']:.4f} sam {rec['sam']:.3f} "
                f"edg {rec['edge']:.4f} con {rec['cons']:.4f}) | val PSNR real {ev['real_psnr']:.2f} "
                f"(bicubic {ev['real_bicubic_psnr']:.2f}) wald {ev['wald_psnr']:.2f} (bicubic {ev['wald_bicubic_psnr']:.2f}) "
                f"SSIM {ev['real_ssim']:.3f} SAM {ev['real_sam']:.2f} | lr {rec['lr']:.1e} | best {best:.2f} "
                f"{'*' if improved else f'({bad}/{args.patience})'} | {tr_time:.0f}s train, {time.time() - t0:.0f}s total | {hours:.2f} h")
            stop_flag[0], stop_flag[1] = float(stop), float(improved)
        if ws > 1:
            dist.broadcast(stop_flag, src=0)
        if stop_flag[0].item() > 0:
            why = "time budget" if ((time.time() - T_START) / 3600 >= args.max_hours) else "early stopping"
            log(f"[stop] {why} at epoch {epoch}")
            break
    return hist


def write_history(out, hist):
    keys = []
    for r in hist:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "metrics_per_epoch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in hist:
            w.writerow(r)


def final_report(args, store, device):
    out = Path(args.out)
    amp = device.type == "cuda" and not args.no_amp
    ck = torch.load(out / "best.pt", map_location=device)
    model = TerraLens(**ck["cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    harm = Harmonizer(ck["stats"], device)
    res = evaluate(model, store, "test", harm, args, device, amp)
    res["best_epoch"] = ck["epoch"]
    res["n_test_tiles"] = store.n("test")
    json.dump(res, open(out / "test_metrics.json", "w"), indent=1)
    log("=" * 78)
    log(f"TEST ({store.n('test')} held-out tiles, EMA weights from epoch {ck['epoch']})")
    for leg in ("real", "wald"):
        log(f"  [{leg:4s}] PSNR {res[f'{leg}_psnr']:.2f} dB (bicubic {res[f'{leg}_bicubic_psnr']:.2f}, "
            f"gain {res[f'{leg}_psnr'] - res[f'{leg}_bicubic_psnr']:+.2f}) | SSIM {res[f'{leg}_ssim']:.4f} "
            f"(bic {res[f'{leg}_bicubic_ssim']:.4f}) | SAM {res[f'{leg}_sam']:.2f} deg (bic {res[f'{leg}_bicubic_sam']:.2f}) | "
            f"ERGAS {res[f'{leg}_ergas']:.3f} | NDVI-MAE {res[f'{leg}_ndvi_mae']:.4f}")
    log(f"  full-resolution consistency  L1(degrade(SR), S2) = {res['consistency_L1']:.5f}")
    log("=" * 78)
    try:
        make_figures(args, store, model, harm, device, amp, out)
    except Exception as e:                                    # noqa
        log("[warn] figure generation failed:", repr(e))
    return res


def make_figures(args, store, model, harm, device, amp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv
    rows = list(_csv.DictReader(open(out / "metrics_per_epoch.csv")))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(ep, [float(r["loss"]) for r in rows]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    for k, lab in (("val_real_psnr", "real"), ("val_wald_psnr", "wald"), ("val_real_bicubic_psnr", "real bicubic"),
                   ("val_wald_bicubic_psnr", "wald bicubic")):
        ax[1].plot(ep, [float(r[k]) for r in rows], "--" if "bicubic" in k else "-", label=lab)
    ax[1].set_title("val PSNR (dB)"); ax[1].legend(); ax[1].set_xlabel("epoch")
    ax[2].plot(ep, [float(r["val_real_sam"]) for r in rows]); ax[2].set_title("val SAM (deg)"); ax[2].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(out / "curves.png", dpi=130); plt.close()
    a, b = store.ranges["test"]
    idx = np.linspace(a, b - 1, num=min(4, b - a)).astype(int)
    lr, hr = store.get(idx, device, harm)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        sr, aux = model(lr, return_aux=True)
    sr = sr.float().clamp(0, 1)
    bic = F.interpolate(lr, scale_factor=args.scale, mode="bicubic", align_corners=False).clamp(0, 1)
    def rgb(t):
        x = t[[2, 1, 0]].permute(1, 2, 0).float().cpu().numpy()
        return np.clip(x / max(np.percentile(x, 99), 1e-3), 0, 1)
    n = len(idx)
    fig, ax = plt.subplots(n, 5, figsize=(16, 3.2 * n))
    ax = np.atleast_2d(ax)
    for i in range(n):
        for j, (im, t) in enumerate([(rgb(bic[i]), "S2 bicubic"), (rgb(sr[i]), "Terra Lens SR"), (rgb(hr[i]), "NAIP (harmonised)")]):
            ax[i, j].imshow(im); ax[i, j].set_title(t); ax[i, j].axis("off")
        ax[i, 3].imshow(aux["gate"][i, 0].float().cpu(), cmap="viridis"); ax[i, 3].set_title("fusion gate g"); ax[i, 3].axis("off")
        ax[i, 4].imshow(aux["edge"][i, 0].float().cpu(), cmap="magma"); ax[i, 4].set_title("edge prior E"); ax[i, 4].axis("off")
    plt.tight_layout(); plt.savefig(out / "samples.png", dpi=110); plt.close()


def make_zip(out: Path, name):
    zp = out.parent / name
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix != ".tmp":
                z.write(p, p.relative_to(out))
    return zp


# =====================================================================================
# 9. Main
# =====================================================================================
def main(argv=None):
    global RANK, LOGF, T_START
    args = get_args(argv)
    T_START = time.time()
    rank = int(os.environ.get("RANK", "0"))
    RANK = rank
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        LOGF = open(out / "train_log.txt", "a")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)

    if args.inspect_only:
        root = find_data_root(args)
        pairs, st, files = discover_pairs(root, args)
        print_inspection(root, pairs, st, files)
        if not pairs:
            log("NO PAIRS. First files under root:\n  " + "\n  ".join(tree_dump(root)))
            return
        p = pairs[0]
        hp = [int(x) for x in args.hr_band_order.split(",")]
        lp = [int(x) for x in args.lr_band_order.split(",")]
        lr, T = load_lr(p["lr"], lp, args.s2_mode)
        hr_raw = to_chw(read_array(p["hr"]))
        log(f"first pair key={p['key']}: S2 composite {lr.shape} from {T} revisit(s), range [{float(lr.min()):.3f},{float(lr.max()):.3f}]"
            f" | NAIP raw {hr_raw.shape} {hr_raw.dtype} range [{hr_raw.min()},{hr_raw.max()}]")
        log(f"implied ratio NAIP/S2 = {hr_raw.shape[-1] / lr.shape[-1]:.2f}x -> resized to {args.scale}x with area averaging")
        return

    if args.benchmark:
        r, ws, lrank, dev = setup_dist()
        run_benchmark(args, r, ws, lrank, dev)
        return

    cache_dir = default_cache_dir(args)
    flag = cache_dir / "READY"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if args.rebuild_cache and flag.exists():
            flag.unlink()
        if not flag.exists() or args.rebuild_cache:
            build_cache(args, cache_dir, find_data_root(args))
            flag.write_text("ok")
        else:
            m = json.load(open(cache_dir / "meta.json"))
            log(f"[cache] found existing cache ({m['n']} tiles) at {cache_dir}; use --rebuild_cache to redo")
    else:
        wait_for(flag)
    rank, ws, lrank, device = setup_dist()
    store = TileStore(cache_dir)
    if rank == 0:
        json.dump({"args": vars(args), "world_size": ws, "device": str(device),
                   "torch": torch.__version__, "cache_meta": {k: store.meta[k] for k in
                   ("n", "counts", "stats", "revisits_median", "diagnostics")}}, open(out / "config.json", "w"), indent=1, default=str)
        json.dump(store.stats, open(out / "stats.json", "w"), indent=1)
    hist = train(args, store, rank, ws, lrank, device)
    if ws > 1:
        dist.barrier()
    if rank == 0:
        final_report(args, store, device)
        zp = make_zip(out, args.zip_name)
        log(f"[done] total {(time.time() - T_START) / 3600:.2f} h | results zip: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
