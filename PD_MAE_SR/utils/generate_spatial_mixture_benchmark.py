"""Generate a deterministic x4 benchmark with spatially mixed degradations.

Each HR image is degraded twice with matched random draws at a mild and a hard
quality level.  The two aligned LR observations are blended with deterministic
soft spatial masks.  The mixtures are unseen during downstream training and
test whether a predicted local restoration-demand field remains useful when
degradation severity varies within one image.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEGRADATION_SOURCE = (
    REPOSITORY_ROOT / 'tools' / 'degradation' /
    'RealisticDegradationGenerator_RealESRGAN.py'
)
GENERATOR_SOURCE = Path(__file__).resolve()


def load_degradation_generator():
    spec = importlib.util.spec_from_file_location(
        'journal_realistic_degradation', DEGRADATION_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealisticDegradationGenerator


def parse_dataset(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('dataset must use NAME=/absolute/path')
    name, raw_path = value.split('=', 1)
    path = Path(raw_path).expanduser().resolve()
    if not name.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f'Invalid dataset specification: {value}')
    return name.strip(), path


def image_paths(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def stable_seed(base_seed, dataset, filename):
    token = f'{base_seed}|{dataset}|{filename}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'big')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
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


def spatial_mask(pattern, height, width, seed):
    if pattern == 'linear':
        return np.broadcast_to(
            np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :],
            (height, width),
        ).copy()
    if pattern == 'radial':
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        yy = (yy - (height - 1) / 2.0) / max(height / 2.0, 1.0)
        xx = (xx - (width - 1) / 2.0) / max(width / 2.0, 1.0)
        return np.clip(np.sqrt(xx ** 2 + yy ** 2), 0.0, 1.0)
    if pattern == 'random_field':
        rng = np.random.default_rng(seed)
        coarse_height = max(2, int(np.ceil(height / 24)))
        coarse_width = max(2, int(np.ceil(width / 24)))
        coarse = rng.uniform(0.0, 1.0, (coarse_height, coarse_width)).astype(np.float32)
        field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        sigma = max(1.0, min(height, width) / 32.0)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        minimum, maximum = float(field.min()), float(field.max())
        return (field - minimum) / max(maximum - minimum, 1e-8)
    raise ValueError(f'Unknown spatial pattern: {pattern}.')


def degrade(image, level, seed, generator_class):
    generator = generator_class()
    generator.set_level(level)
    random.seed(seed)
    np.random.seed(seed)
    return generator.apply_degradation(image, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', required=True, type=parse_dataset)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--mild_level', type=int, default=90)
    parser.add_argument('--hard_level', type=int, default=60)
    parser.add_argument(
        '--patterns', nargs='+', default=['linear', 'radial', 'random_field'],
        choices=['linear', 'radial', 'random_field'],
    )
    parser.add_argument('--seed', type=int, default=276)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    generator_class = load_degradation_generator()
    supported = generator_class().degradation_levels
    if args.mild_level not in supported or args.hard_level not in supported:
        raise ValueError(f'Levels must be selected from {sorted(supported)}.')
    if args.mild_level == args.hard_level:
        raise ValueError('Mild and hard levels must differ.')

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 2,
        'repository_commit': repository_commit(),
        'base_seed': args.seed,
        'scale': 4,
        'mild_level': args.mild_level,
        'hard_level': args.hard_level,
        'patterns': args.patterns,
        'datasets': {name: str(path) for name, path in args.dataset},
        'degradation_source': str(DEGRADATION_SOURCE),
        'degradation_source_sha256': sha256(DEGRADATION_SOURCE),
        'generator_source': str(GENERATOR_SOURCE),
        'generator_source_sha256': sha256(GENERATOR_SOURCE),
        'numpy_version': np.__version__,
        'opencv_version': cv2.__version__,
        'max_images_per_dataset': args.max_images,
        'alignment': 'top-left HR crop to dimensions divisible by four',
        'mixture': 'LQ = (1-alpha)*mild + alpha*hard',
    }
    protocol_path = output_root / 'protocol.json'
    if protocol_path.exists() and not args.overwrite:
        existing = json.loads(protocol_path.read_text(encoding='utf-8'))
        if existing != protocol:
            raise RuntimeError(
                f'{protocol_path} belongs to a different protocol. '
                'Choose another output root or pass --overwrite.'
            )
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    manifest = []
    for dataset_name, source_root in args.dataset:
        paths = image_paths(source_root)
        if args.max_images > 0:
            paths = paths[:args.max_images]
        if not paths:
            raise RuntimeError(f'No images found in {source_root}.')
        stems = [path.stem for path in paths]
        if len(stems) != len(set(stems)):
            raise RuntimeError(f'Filename stems are not unique in {source_root}.')

        for index, source_path in enumerate(paths, start=1):
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot read {source_path}.')
            height, width = image.shape[:2]
            aligned_height = height - height % 4
            aligned_width = width - width % 4
            if min(aligned_height, aligned_width) < 4:
                raise RuntimeError(f'Image is too small for x4 SR: {source_path}.')
            gt = image[:aligned_height, :aligned_width]
            filename = f'{source_path.stem}.png'
            gt_path = output_root / dataset_name / 'GT' / filename
            if args.overwrite or not gt_path.exists():
                write_png(gt_path, gt)

            seed = stable_seed(args.seed, dataset_name, source_path.name)
            mild = degrade(gt, args.mild_level, seed, generator_class)
            hard = degrade(gt, args.hard_level, seed, generator_class)
            expected_shape = (aligned_height // 4, aligned_width // 4)
            if mild.shape[:2] != expected_shape or hard.shape[:2] != expected_shape:
                raise RuntimeError(
                    f'Unexpected degraded shape for {source_path}: '
                    f'{mild.shape[:2]}, {hard.shape[:2]} vs {expected_shape}.'
                )
            mild_path = (
                output_root / dataset_name / f'q{args.mild_level}' / filename
            )
            hard_path = (
                output_root / dataset_name / f'q{args.hard_level}' / filename
            )
            for path, degraded in ((mild_path, mild), (hard_path, hard)):
                if args.overwrite or not path.exists():
                    write_png(path, degraded)

            shared_hashes = {
                'source_sha256': sha256(source_path),
                'gt_sha256': sha256(gt_path),
                'mild_sha256': sha256(mild_path),
                'hard_sha256': sha256(hard_path),
            }

            for pattern_index, pattern in enumerate(args.patterns):
                alpha = spatial_mask(
                    pattern,
                    expected_shape[0],
                    expected_shape[1],
                    seed + 1009 * (pattern_index + 1),
                )
                mixture = np.rint(
                    mild.astype(np.float32) * (1.0 - alpha[..., None])
                    + hard.astype(np.float32) * alpha[..., None]
                ).clip(0, 255).astype(np.uint8)
                mixture_path = output_root / dataset_name / f'mix_{pattern}' / filename
                mask_path = output_root / dataset_name / 'masks' / pattern / filename
                if args.overwrite or not mixture_path.exists():
                    write_png(mixture_path, mixture)
                if args.overwrite or not mask_path.exists():
                    write_png(mask_path, np.rint(alpha * 255.0).astype(np.uint8))
                manifest.append({
                    'dataset': dataset_name,
                    'source_name': source_path.name,
                    'filename': filename,
                    'pattern': pattern,
                    'seed': seed,
                    'gt': gt_path.relative_to(output_root).as_posix(),
                    'mild_lq': mild_path.relative_to(output_root).as_posix(),
                    'hard_lq': hard_path.relative_to(output_root).as_posix(),
                    'lq': mixture_path.relative_to(output_root).as_posix(),
                    'mask': mask_path.relative_to(output_root).as_posix(),
                    **shared_hashes,
                    'lq_sha256': sha256(mixture_path),
                    'mask_sha256': sha256(mask_path),
                    'gt_height': aligned_height,
                    'gt_width': aligned_width,
                })
            if index % 10 == 0 or index == len(paths):
                print(f'{dataset_name}: generated {index}/{len(paths)}', flush=True)

    manifest_path = output_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary_path = output_root / 'manifest_summary.json'
    summary_path.write_text(json.dumps({
        'schema_version': 1,
        'manifest': manifest_path.name,
        'manifest_sha256': sha256(manifest_path),
        'records': len(manifest),
        'unique_source_images': len({
            (row['dataset'], row['source_name']) for row in manifest
        }),
        'patterns': args.patterns,
        'protocol_sha256': sha256(protocol_path),
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(manifest)} spatial mixtures to {output_root}.')


if __name__ == '__main__':
    main()
