"""Evaluate arbitrary SR output folders with one fixed BasicSR protocol.

This evaluator is intentionally model-agnostic.  Outputs produced by this
repository or by an external baseline pass through the same filename, shape,
metric, and provenance checks before they can enter a reproduced comparison
table.  Published numbers evaluated under a different protocol must not be
mixed with these results.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
METRIC_PROTOCOL = {
    'psnr': {'reference': True, 'better': 'higher', 'color_space': 'y'},
    'ssim': {'reference': True, 'better': 'higher', 'color_space': 'y'},
    'niqe': {'reference': False, 'better': 'lower', 'color_space': 'y'},
}


def repository_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_index(root, suffix=''):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f'Image directory does not exist: {root}')
    index = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if suffix:
            if not stem.endswith(suffix):
                raise RuntimeError(
                    f'Image stem {stem!r} does not end with suffix {suffix!r}.'
                )
            stem = stem[:-len(suffix)]
        if stem in index:
            raise RuntimeError(
                f'Duplicate normalized image stem {stem!r} in {root}: '
                f'{index[stem]} and {path}'
            )
        index[stem] = path
    if not index:
        raise RuntimeError(f'No images found in {root}.')
    return index


def aligned_triplets(sr_root, gt_root, lq_root=None, sr_suffix='', lq_suffix=''):
    """Return exactly aligned SR/GT/(optional) LQ paths, indexed by stem."""
    indices = {
        'sr': image_index(sr_root, sr_suffix),
        'gt': image_index(gt_root),
    }
    if lq_root is not None:
        indices['lq'] = image_index(lq_root, lq_suffix)

    reference_names = set(indices['gt'])
    errors = []
    for role, index in indices.items():
        names = set(index)
        missing = sorted(reference_names - names)
        extra = sorted(names - reference_names)
        if missing or extra:
            errors.append(
                f'{role}: missing={len(missing)} extra={len(extra)} '
                f'examples={(missing + extra)[:5]}'
            )
    if errors:
        raise RuntimeError('Image sets are not identical: ' + '; '.join(errors))

    return [
        (
            stem,
            indices['sr'][stem],
            indices['gt'][stem],
            indices.get('lq', {}).get(stem),
        )
        for stem in sorted(reference_names)
    ]


def verify_shapes(stem, sr_shape, gt_shape, lq_shape=None, scale=4):
    if tuple(sr_shape) != tuple(gt_shape):
        raise RuntimeError(
            f'SR/GT shape mismatch for {stem}: SR={sr_shape}, GT={gt_shape}.'
        )
    if lq_shape is not None:
        expected = (lq_shape[0] * scale, lq_shape[1] * scale)
        if tuple(sr_shape[:2]) != expected:
            raise RuntimeError(
                f'Output scale mismatch for {stem}: SR={sr_shape[:2]}, '
                f'LQ={lq_shape[:2]}, expected SR={expected} at scale {scale}.'
            )


def image_set_digest(items):
    """Hash image identities and bytes without depending on absolute paths."""
    digest = hashlib.sha256()
    for stem, path in sorted(items):
        path = Path(path)
        digest.update(stem.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(path.stat().st_size).encode('ascii'))
        digest.update(b'\0')
        digest.update(file_sha256(path).encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


def summarize(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError('Metric output contains no values or non-finite values.')
    return {
        'images': len(values),
        'mean': statistics.fmean(values),
        'sample_std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'minimum': min(values),
        'maximum': max(values),
    }


def validate_metric_row(stem, metrics):
    non_finite = {
        name: float(value)
        for name, value in metrics.items()
        if not math.isfinite(float(value))
    }
    if non_finite:
        raise RuntimeError(
            f'Non-finite metric output for image {stem!r}: {non_finite}.'
        )


def optional_distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sr_root', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--lq_root')
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--sr_suffix', default='')
    parser.add_argument('--lq_suffix', default='')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--crop_border', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scale <= 0:
        raise ValueError('--scale must be positive.')
    if args.crop_border < 0:
        raise ValueError('--crop_border must be non-negative.')

    import cv2
    import numpy as np
    from basicsr.metrics import calculate_niqe, calculate_psnr, calculate_ssim

    pairs = aligned_triplets(
        args.sr_root,
        args.gt_root,
        args.lq_root,
        args.sr_suffix,
        args.lq_suffix,
    )
    rows = []
    for stem, sr_path, gt_path, lq_path in pairs:
        sr = cv2.imread(str(sr_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR) if lq_path else None
        if sr is None or gt is None or (lq_path and lq is None):
            raise RuntimeError(f'Failed to decode one or more images for {stem}.')
        verify_shapes(
            stem,
            sr.shape,
            gt.shape,
            lq.shape if lq is not None else None,
            args.scale,
        )
        metrics = {
            'psnr': float(calculate_psnr(
                sr, gt, args.crop_border, input_order='HWC', test_y_channel=True
            )),
            'ssim': float(calculate_ssim(
                sr, gt, args.crop_border, input_order='HWC', test_y_channel=True
            )),
            'niqe': float(calculate_niqe(
                sr, args.crop_border, input_order='HWC', convert_to='y'
            )),
        }
        validate_metric_row(stem, metrics)
        rows.append({'image': stem, **metrics})

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise RuntimeError(f'Checkpoint does not exist: {checkpoint}')
    index_by_role = {
        'sr': [(stem, sr_path) for stem, sr_path, _, _ in pairs],
        'gt': [(stem, gt_path) for stem, _, gt_path, _ in pairs],
    }
    if args.lq_root:
        index_by_role['lq'] = [
            (stem, lq_path) for stem, _, _, lq_path in pairs
        ]
    metric_sources = {
        'psnr_ssim': Path(inspect.getfile(calculate_psnr)).resolve(),
        'niqe': Path(inspect.getfile(calculate_niqe)).resolve(),
    }
    report = {
        'schema_version': 1,
        'method': args.method,
        'dataset': args.dataset,
        'arguments': vars(args),
        'repository_commit': repository_commit(),
        'software': {
            'basicsr': optional_distribution_version('basicsr'),
            'opencv_python': optional_distribution_version('opencv-python'),
            'numpy': np.__version__,
        },
        'metric_sources': {
            name: {'path': str(path), 'sha256': file_sha256(path)}
            for name, path in metric_sources.items()
        },
        'metric_protocol': {
            'crop_border': args.crop_border,
            'scale': args.scale,
            **METRIC_PROTOCOL,
        },
        'artifact_sha256': {
            role: image_set_digest(items) for role, items in index_by_role.items()
        },
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': file_sha256(checkpoint) if checkpoint else None,
        'metrics': {
            name: summarize(row[name] for row in rows)
            for name in METRIC_PROTOCOL
        },
        'per_image_csv': str(output_csv),
    }
    with output_json.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
