"""Measure fidelity separately in mild and hard regions of a mixed benchmark."""

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def find_sr(sr_root, stem, suffix):
    if suffix:
        candidates = [sr_root / f'{stem}_{suffix}{extension}' for extension in IMAGE_SUFFIXES]
    else:
        candidates = [path for path in sr_root.glob(f'{stem}*') if path.suffix.lower() in IMAGE_SUFFIXES]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one SR result for {stem}, found {matches}.')
    return matches[0]


def weighted_psnr(sr, gt, weight):
    squared_error = np.mean((sr - gt) ** 2, axis=2)
    mse = float(np.sum(squared_error * weight) / max(np.sum(weight), 1e-12))
    return math.inf if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sr_root', type=Path, required=True)
    parser.add_argument('--gt_root', type=Path, required=True)
    parser.add_argument('--mask_root', type=Path, required=True)
    parser.add_argument('--sr_suffix', default='')
    parser.add_argument('--crop_border', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    gt_paths = sorted(
        path for path in args.gt_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not gt_paths:
        raise RuntimeError(f'No GT images found in {args.gt_root}.')

    rows = []
    border = args.crop_border
    for gt_path in gt_paths:
        sr_path = find_sr(args.sr_root, gt_path.stem, args.sr_suffix)
        mask_path = args.mask_root / f'{gt_path.stem}.png'
        sr = cv2.imread(str(sr_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if sr is None or gt is None or mask is None:
            raise RuntimeError(f'Failed to load SR/GT/mask for {gt_path.stem}.')
        if sr.shape != gt.shape:
            raise RuntimeError(f'Shape mismatch for {gt_path.stem}: {sr.shape} vs {gt.shape}.')
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        hard_weight = mask.astype(np.float64) / 255.0
        mild_weight = 1.0 - hard_weight
        sr = sr.astype(np.float64)
        gt = gt.astype(np.float64)
        if border:
            slices = (slice(border, -border), slice(border, -border))
            sr, gt = sr[slices], gt[slices]
            hard_weight = hard_weight[slices]
            mild_weight = mild_weight[slices]
        rows.append({
            'image': gt_path.stem,
            'psnr_all': weighted_psnr(sr, gt, np.ones_like(hard_weight)),
            'psnr_mild': weighted_psnr(sr, gt, mild_weight),
            'psnr_hard': weighted_psnr(sr, gt, hard_weight),
            'hard_fraction': float(hard_weight.mean()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ('psnr_all', 'psnr_mild', 'psnr_hard', 'hard_fraction')
    }
    print({'images': len(rows), **means})
    print(f'Saved per-image regional metrics to {args.output}.')


if __name__ == '__main__':
    main()
