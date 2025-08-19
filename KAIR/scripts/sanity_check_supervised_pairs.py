#!/usr/bin/env python3
import os, random, h5py, numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr

def sample_pairs(root, k=5):
    root = Path(root)
    cases = [p for p in (root).glob('*') if p.is_dir()]
    picks = []
    for case in random.sample(cases, min(len(cases), k)):
        gt = case / 'gt.h5'
        slice_files = sorted([p for p in case.glob('*.h5') if p.name != 'gt.h5'])
        if not slice_files or not gt.exists():
            continue
        for s in random.sample(slice_files, min(len(slice_files), 2)):
            picks.append((s, gt))
    return picks

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-root', required=True, help='Path to /train folder of supervised dataset')
    args = ap.parse_args()

    pairs = sample_pairs(args.train_root, k=3)
    assert pairs, f'No pairs found under {args.train_root}'

    for s_path, gt_path in pairs:
        with h5py.File(s_path, 'r') as fs, h5py.File(gt_path, 'r') as fg:
            assert 'LI_CT' in fs, f"LI_CT missing in {s_path} keys={list(fs.keys())}"
            assert 'image' in fg, f"image missing in {gt_path} keys={list(fg.keys())}"
            L = fs['LI_CT'][:].astype(np.float32)
            H = fg['image'][:].astype(np.float32)
        print(f"Pair: {s_path.relative_to(args.train_root)} | L/H shapes: {L.shape}/{H.shape} | ranges: L[{L.min():.1f},{L.max():.1f}], H[{H.min():.1f},{H.max():.1f}]")
        # Expect 0..255 float
        if not (0.0 <= L.min() and L.max() <= 255.0 and 0.0 <= H.min() and H.max() <= 255.0):
            print('  ! Range check failed (expected 0..255 floats).')
        # Quick PSNR of input vs gt in 0..255
        p = psnr(H, L, data_range=255.0)
        print(f"  PSNR(L->H) [0-255]: {p:.2f} dB (expect ~15-30dB depending on artifact strength)")
