"""Paired partial-degradation crops with source-disjoint splitting."""

import glob
import os
import random
import re
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PartialRestorationDataset(Dataset):
    """Load the exact partial-degradation observation and its clean target.

    Every root is expected to contain ``PD_LQ`` and ``HR`` subdirectories.
    Unlike the legacy Stage-1 loader, this dataset does not recompute a binary
    complexity mask. Supervision can therefore be derived from the actual
    paired correction, including feathered transition regions.
    """

    def __init__(self, roots, patch_size=256, augment=True, split='all',
                 val_fraction=0.05, split_seed=0,
                 deterministic_crop=False):
        if isinstance(roots, str):
            roots = [roots]
        if not roots:
            raise ValueError('At least one partial-degradation root is required.')
        if split not in {'all', 'train', 'val'}:
            raise ValueError(f'Unsupported split: {split}')
        if split != 'all' and not 0.0 < val_fraction < 1.0:
            raise ValueError('val_fraction must be in (0, 1) for a split dataset.')

        self.patch_size = patch_size
        self.augment = augment
        self.deterministic_crop = deterministic_crop
        self.samples = []
        for raw_root in roots:
            root = os.path.expanduser(raw_root)
            lq_root = os.path.join(root, 'PD_LQ')
            hr_root = os.path.join(root, 'HR')
            lq_paths = []
            for extension in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
                lq_paths.extend(glob.glob(os.path.join(lq_root, extension)))
            for lq_path in sorted(lq_paths):
                filename = os.path.basename(lq_path)
                hr_path = os.path.join(hr_root, filename)
                if not os.path.isfile(hr_path):
                    continue
                self.samples.append({
                    'lq_path': lq_path,
                    'hr_path': hr_path,
                    'filename': filename,
                    'root': os.path.basename(os.path.normpath(root)),
                    'source_group': self.source_group(filename),
                })
        if not self.samples:
            raise RuntimeError(f'No paired partial-degradation images found in {roots}.')

        if split != 'all':
            grouped = {}
            for sample in self.samples:
                grouped.setdefault(sample['source_group'], []).append(sample)
            group_names = sorted(grouped)
            random.Random(split_seed).shuffle(group_names)
            target_val = max(1, int(round(len(self.samples) * val_fraction)))
            val_groups = set()
            val_images = 0
            for group in group_names:
                if val_images >= target_val:
                    break
                val_groups.add(group)
                val_images += len(grouped[group])
            want_validation = split == 'val'
            self.samples = [
                sample for sample in self.samples
                if (sample['source_group'] in val_groups) == want_validation
            ]
        if not self.samples:
            raise RuntimeError(
                f'The {split} partition is empty; use more source groups or '
                'adjust val_fraction.'
            )
        self.source_groups = sorted({
            sample['source_group'] for sample in self.samples
        })
        self.root_counts = Counter(sample['root'] for sample in self.samples)

    def root_balancing_weights(self):
        """Return per-sample weights that assign equal mass to every root."""
        return [
            1.0 / self.root_counts[sample['root']] for sample in self.samples
        ]

    @staticmethod
    def source_group(filename):
        stem = os.path.splitext(os.path.basename(filename))[0]
        match = re.match(r'^(.*)_p\d+_l\d+_\d+$', stem)
        return match.group(1) if match else stem

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        lq = cv2.imread(sample['lq_path'], cv2.IMREAD_COLOR)
        hr = cv2.imread(sample['hr_path'], cv2.IMREAD_COLOR)
        if lq is None or hr is None:
            raise RuntimeError(f'Could not load pair: {sample}')
        if lq.shape[:2] != hr.shape[:2]:
            raise RuntimeError(
                f'Unaligned partial pair {sample["filename"]}: '
                f'{lq.shape[:2]} vs {hr.shape[:2]}.'
            )

        height, width = hr.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            hr = cv2.resize(
                hr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC
            )
            lq = cv2.resize(
                lq, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC
            )
        else:
            if self.deterministic_crop:
                top = (height - self.patch_size) // 2
                left = (width - self.patch_size) // 2
            else:
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            hr = hr[top:top + self.patch_size, left:left + self.patch_size]
            lq = lq[top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                hr = np.flip(hr, axis=1)
                lq = np.flip(lq, axis=1)
            rotation = random.randrange(4)
            if rotation:
                hr = np.rot90(hr, rotation)
                lq = np.rot90(lq, rotation)

        hr = cv2.cvtColor(np.ascontiguousarray(hr), cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(np.ascontiguousarray(lq), cv2.COLOR_BGR2RGB)
        hr = torch.from_numpy(hr).permute(2, 0, 1).float().div_(255.0)
        lq = torch.from_numpy(lq).permute(2, 0, 1).float().div_(255.0)
        return {
            'lq': lq,
            'hr': hr,
            'filename': sample['filename'],
            'root': sample['root'],
            'source_group': sample['source_group'],
        }
