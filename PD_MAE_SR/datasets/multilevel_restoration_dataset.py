import glob
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ALIGNMENT_BACKENDS = {'opencv_bicubic', 'pytorch_bicubic'}


def align_lq_to_hr(lq, height, width, backend):
    """Align a decoded BGR LQ image to HR size without re-quantization."""
    if lq.shape[:2] == (height, width):
        return lq
    if backend == 'opencv_bicubic':
        return cv2.resize(lq, (width, height), interpolation=cv2.INTER_CUBIC)
    if backend != 'pytorch_bicubic':
        raise ValueError(
            f'Unsupported alignment backend {backend!r}; '
            f'expected one of {sorted(ALIGNMENT_BACKENDS)}.'
        )
    tensor = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float().div_(255.0)
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode='bicubic',
        align_corners=False,
    ).clamp_(0.0, 1.0)
    return resized.squeeze(0).permute(1, 2, 0).numpy()


class MultiLevelRestorationDataset(Dataset):
    """Aligned HR/LQ crops sampled from several degradation levels.

    Every LQ root must use the same filenames as ``hr_root``.  A degradation
    level is sampled for each training item, which exposes the restoration-need
    predictor to both mild and severe full-image corruptions.
    """

    def __init__(self, hr_root, lq_roots, patch_size=256, augment=True,
                 deterministic_level=False, split='all', val_fraction=0.05,
                 split_seed=0, deterministic_crop=False,
                 split_unit='filename',
                 alignment_backend='opencv_bicubic'):
        self.hr_root = os.path.expanduser(hr_root)
        self.lq_roots = [os.path.expanduser(path) for path in lq_roots]
        self.patch_size = patch_size
        self.augment = augment
        self.deterministic_level = deterministic_level
        self.deterministic_crop = deterministic_crop
        if alignment_backend not in ALIGNMENT_BACKENDS:
            raise ValueError(
                f'Unsupported alignment backend {alignment_backend!r}; '
                f'expected one of {sorted(ALIGNMENT_BACKENDS)}.'
            )
        self.alignment_backend = alignment_backend

        if not self.lq_roots:
            raise ValueError('At least one LQ root is required.')

        extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        self.hr_paths = []
        for extension in extensions:
            self.hr_paths.extend(glob.glob(os.path.join(self.hr_root, extension)))
        self.hr_paths = sorted(self.hr_paths)
        if not self.hr_paths:
            raise RuntimeError(f'No HR images found in {self.hr_root}.')

        if split not in {'all', 'train', 'val'}:
            raise ValueError(f'Unsupported split: {split}')
        if split_unit not in {'filename', 'source'}:
            raise ValueError(f'Unsupported split unit: {split_unit}')
        self.split_unit = split_unit
        if split != 'all':
            if not 0.0 < val_fraction < 1.0:
                raise ValueError('val_fraction must be in (0, 1) for a split dataset.')
            if split_unit == 'filename':
                indices = list(range(len(self.hr_paths)))
                random.Random(split_seed).shuffle(indices)
                num_val = max(1, int(round(len(indices) * val_fraction)))
                selected = indices[:num_val] if split == 'val' else indices[num_val:]
                self.hr_paths = [self.hr_paths[index] for index in selected]
            else:
                grouped = {}
                for path in self.hr_paths:
                    group = self.source_group(os.path.basename(path))
                    grouped.setdefault(group, []).append(path)
                group_names = sorted(grouped)
                random.Random(split_seed).shuffle(group_names)
                target_val = max(1, int(round(len(self.hr_paths) * val_fraction)))
                val_groups = set()
                val_images = 0
                for group in group_names:
                    if val_images >= target_val:
                        break
                    val_groups.add(group)
                    val_images += len(grouped[group])
                want_validation = split == 'val'
                self.hr_paths = [
                    path for path in self.hr_paths
                    if (self.source_group(os.path.basename(path)) in val_groups)
                    == want_validation
                ]

        if not self.hr_paths:
            raise RuntimeError(
                f'The {split} partition is empty; use more source groups or '
                'adjust val_fraction.'
            )

        self.source_groups = sorted({
            self.source_group(os.path.basename(path)) for path in self.hr_paths
        })

        required_names = [os.path.basename(path) for path in self.hr_paths]
        for root in self.lq_roots:
            missing = [
                name for name in required_names
                if not os.path.isfile(os.path.join(root, name))
            ]
            if missing:
                raise RuntimeError(
                    f'LQ root {root} is missing {len(missing)} files required '
                    f'by the {split} partition; examples={missing[:3]}.'
                )

    @staticmethod
    def source_group(filename):
        """Map an offline crop name such as ``000617_s014`` to its source."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        match = re.match(r'^(.*)_s\d+$', stem)
        return match.group(1) if match else stem

    def __len__(self):
        return len(self.hr_paths)

    def _level_index(self, index):
        if self.deterministic_level:
            return index % len(self.lq_roots)
        return random.randrange(len(self.lq_roots))

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        filename = os.path.basename(hr_path)
        level_index = self._level_index(index)
        lq_path = os.path.join(self.lq_roots[level_index], filename)

        hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        if hr is None:
            raise RuntimeError(f'Could not read HR image: {hr_path}')
        if lq is None:
            raise RuntimeError(f'Could not read paired LQ image: {lq_path}')

        height, width = hr.shape[:2]
        lq = align_lq_to_hr(
            lq, height, width, backend=self.alignment_backend
        )

        if height < self.patch_size or width < self.patch_size:
            hr = cv2.resize(hr, (self.patch_size, self.patch_size),
                            interpolation=cv2.INTER_CUBIC)
            lq = cv2.resize(lq, (self.patch_size, self.patch_size),
                            interpolation=cv2.INTER_CUBIC)
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
        lq_is_integer = np.issubdtype(lq.dtype, np.integer)
        hr = torch.from_numpy(hr).permute(2, 0, 1).float().div_(255.0)
        lq = torch.from_numpy(lq).permute(2, 0, 1).float()
        if lq_is_integer:
            lq.div_(255.0)

        return {
            'lq': lq,
            'hr': hr,
            'level_index': torch.tensor(level_index, dtype=torch.long),
            'filename': filename,
            'source_group': self.source_group(filename),
        }
