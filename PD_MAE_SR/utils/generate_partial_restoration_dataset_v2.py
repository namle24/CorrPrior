"""Generate deterministic, manifest-backed partial-degradation pretraining data."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEGRADATION_SOURCE = (
    REPOSITORY_ROOT / 'tools' / 'degradation' /
    'RealisticDegradationGenerator_RealESRGAN.py'
)
MASK_SOURCE = REPOSITORY_ROOT / 'PD_MAE_SR' / 'utils' / 'pd_mae_utils.py'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
_GENERATOR_CLASS = None
_COMPLEXITY_MASK = None


def load_symbol(path, module_name, symbol):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def initialize_worker():
    global _GENERATOR_CLASS, _COMPLEXITY_MASK
    _GENERATOR_CLASS = load_symbol(
        DEGRADATION_SOURCE, 'partial_v2_degradation',
        'RealisticDegradationGenerator'
    )
    _COMPLEXITY_MASK = load_symbol(
        MASK_SOURCE, 'partial_v2_mask', 'compute_complexity_mask'
    )


def parse_dataset(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('dataset must use NAME=/absolute/path')
    name, raw_path = value.split('=', 1)
    path = Path(raw_path).expanduser().resolve()
    if not name.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f'Invalid dataset specification: {value}')
    return name.strip(), path


def stable_seed(*parts):
    token = '|'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'big')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f'Failed to write {path}.')


def resized_for_crop(image, patch_size):
    height, width = image.shape[:2]
    if height >= patch_size and width >= patch_size:
        return image
    factor = max(patch_size / height, patch_size / width)
    target = (int(math.ceil(width * factor)), int(math.ceil(height * factor)))
    return cv2.resize(image, target, interpolation=cv2.INTER_CUBIC)


def process_task(task):
    (source_path, dataset_name, patch_index, output_root, levels,
     patch_size, scale_factor, degrade_ratio, complexity_patch_size,
     feather_kernel, feather_sigma, base_seed, overwrite) = task
    source_path = Path(source_path)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read {source_path}.')
    image = resized_for_crop(image, patch_size)
    height, width = image.shape[:2]
    crop_seed = stable_seed(base_seed, dataset_name, source_path.name, patch_index, 'crop')
    crop_rng = random.Random(crop_seed)
    top = crop_rng.randint(0, height - patch_size)
    left = crop_rng.randint(0, width - patch_size)
    hr = image[top:top + patch_size, left:left + patch_size]

    binary_mask = _COMPLEXITY_MASK(
        hr, degrade_ratio=degrade_ratio, patch_size=complexity_patch_size
    )
    soft_mask = cv2.GaussianBlur(
        binary_mask.astype(np.float32),
        (feather_kernel, feather_kernel),
        feather_sigma,
    )
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]

    degradation_seed = stable_seed(
        base_seed, dataset_name, source_path.name, patch_index, 'degrade'
    )
    rows = []
    for level in levels:
        # Reset both RNGs to the same state for every level. This matches the
        # stochastic draw as far as level-dependent control flow allows, so
        # q90/q80/q70/q60 differ mainly through their configured ranges rather
        # than unrelated random samples.
        random.seed(degradation_seed)
        np.random.seed(degradation_seed)
        generator = _GENERATOR_CLASS()
        generator.set_level(level)
        degraded_lr = generator.apply_degradation(hr, scale_factor)
        degraded = cv2.resize(
            degraded_lr, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )
        partial = np.rint(
            hr.astype(np.float32) * (1.0 - soft_mask)
            + degraded.astype(np.float32) * soft_mask
        ).clip(0, 255).astype(np.uint8)

        filename = (
            f'{source_path.stem}_p{patch_index:03d}_l{level}_'
            f'{degradation_seed:010d}.png'
        )
        root = output_root / dataset_name / f'PD_{level}'
        hr_path = root / 'HR' / filename
        lq_path = root / 'PD_LQ' / filename
        mask_path = root / 'mask' / filename
        if overwrite or not (hr_path.exists() and lq_path.exists() and mask_path.exists()):
            write_png(hr_path, hr)
            write_png(lq_path, partial)
            write_png(mask_path, np.rint(soft_mask[..., 0] * 255.0).astype(np.uint8))
        rows.append({
            'dataset': dataset_name,
            'source_path': str(source_path),
            'source_group': source_path.stem,
            'patch_index': patch_index,
            'crop_seed': crop_seed,
            'degradation_seed': degradation_seed,
            'level': level,
            'crop_top': top,
            'crop_left': left,
            'patch_size': patch_size,
            'filename': filename,
            'hr': hr_path.relative_to(output_root).as_posix(),
            'pd_lq': lq_path.relative_to(output_root).as_posix(),
            'mask': mask_path.relative_to(output_root).as_posix(),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', required=True, type=parse_dataset)
    parser.add_argument('--output_root', required=True, type=Path)
    parser.add_argument('--levels', nargs='+', type=int, default=[90, 80, 70, 60])
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--patches_per_image', type=int, default=1)
    parser.add_argument('--scale_factor', type=float, default=0.25)
    parser.add_argument('--degrade_ratio', type=float, default=0.75)
    parser.add_argument('--complexity_patch_size', type=int, default=16)
    parser.add_argument('--feather_kernel', type=int, default=15)
    parser.add_argument('--feather_sigma', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=276)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.patch_size <= 0 or args.patches_per_image <= 0:
        raise ValueError('Patch size and patches per image must be positive.')
    if (args.complexity_patch_size <= 0
            or args.patch_size % args.complexity_patch_size):
        raise ValueError(
            'complexity_patch_size must be positive and divide patch_size.'
        )
    if not 0.0 < args.scale_factor < 1.0:
        raise ValueError('scale_factor must be in (0, 1).')
    if not 0.0 < args.degrade_ratio < 1.0:
        raise ValueError('degrade_ratio must be in (0, 1).')
    if args.feather_kernel <= 0 or args.feather_kernel % 2 == 0:
        raise ValueError('feather_kernel must be a positive odd integer.')

    initialize_worker()
    supported = set(_GENERATOR_CLASS().degradation_levels)
    if not set(args.levels).issubset(supported):
        raise ValueError(f'Levels must be selected from {sorted(supported)}.')

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        'arguments': {
            **vars(args),
            'output_root': str(output_root),
            'dataset': [(name, str(path)) for name, path in args.dataset],
        },
        'git_commit': repository_commit(),
        'degradation_source': str(DEGRADATION_SOURCE),
        'degradation_source_sha256': sha256(DEGRADATION_SOURCE),
        'mask_source': str(MASK_SOURCE),
        'mask_source_sha256': sha256(MASK_SOURCE),
        'upscaling': 'bicubic LR-to-pretraining-patch alignment',
        'level_pairing': 'matched Python and NumPy RNG seed across levels',
        'mask_selection': (
            'exact top-k Sobel-complexity patches with a deterministic '
            'image-dependent tie break'
        ),
        'mixture': 'PD_LQ = (1-mask)*HR + mask*bicubic(degraded_LR)',
    }
    protocol_path = output_root / 'protocol.json'
    if protocol_path.exists() and not args.overwrite:
        existing = json.loads(protocol_path.read_text(encoding='utf-8'))
        if existing != protocol:
            raise RuntimeError(
                f'{protocol_path} belongs to a different protocol. '
                'Use another output root or pass --overwrite.'
            )
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    tasks = []
    for dataset_name, source_root in args.dataset:
        sources = sorted(
            path for path in source_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if args.max_images > 0:
            sources = sources[:args.max_images]
        if not sources:
            raise RuntimeError(f'No images found in {source_root}.')
        stems = [path.stem for path in sources]
        if len(stems) != len(set(stems)):
            raise RuntimeError(f'Filename stems are not unique in {source_root}.')
        for source_path in sources:
            for patch_index in range(args.patches_per_image):
                tasks.append((
                    source_path, dataset_name, patch_index, output_root,
                    tuple(args.levels), args.patch_size, args.scale_factor,
                    args.degrade_ratio, args.complexity_patch_size,
                    args.feather_kernel, args.feather_sigma, args.seed,
                    args.overwrite,
                ))

    rows = []
    with Pool(args.workers, initializer=initialize_worker) as pool:
        for task_rows in tqdm(pool.imap(process_task, tasks), total=len(tasks)):
            rows.extend(task_rows)
    manifest_path = output_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} paired samples to {output_root}.')


if __name__ == '__main__':
    main()
