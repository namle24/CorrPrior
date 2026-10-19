"""Evaluate paired SR folders with reproducible PyIQA metrics.

The script is deliberately separate from BasicSR validation so perceptual-IQA
dependencies cannot alter the training environment. It writes one row per
aligned image and a provenance-rich JSON summary suitable for paired tests.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
METRIC_PROTOCOL = {
    'lpips': {'reference': True, 'better': 'lower'},
    'dists': {'reference': True, 'better': 'lower'},
    'musiq': {'reference': False, 'better': 'higher'},
    # These two metrics are optional secondary checks.  They are not part of
    # the preregistered DISTS endpoint used to select the proposed model.
    'topiq_fr-pipal': {'reference': True, 'better': 'higher'},
    'clipiqa+': {'reference': False, 'better': 'higher'},
}


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def image_set_digest(items):
    """Hash normalized image identities and bytes, independent of root path."""
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


def package_python_digest(package):
    """Bind a metric report to the imported package's Python implementation."""
    root = Path(package.__file__).resolve().parent
    files = sorted(
        path for path in root.rglob('*.py') if '__pycache__' not in path.parts
    )
    if not files:
        raise RuntimeError(f'No Python source files found under {root}.')
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\n')
    return {
        'root': str(root),
        'sha256': digest.hexdigest(),
        'python_files': len(files),
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


def image_index(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f'Image directory does not exist: {root}')
    index = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            raise RuntimeError(
                f'Duplicate image stem {path.stem!r} in {root}: '
                f'{index[path.stem]} and {path}'
            )
        index[path.stem] = path
    if not index:
        raise RuntimeError(f'No images found in {root}.')
    return index


def aligned_pairs(sr_root, gt_root, sr_suffix=''):
    """Return stem-aligned SR/GT paths and reject incomplete evaluation sets."""
    sr_index = image_index(sr_root)
    gt_index = image_index(gt_root)
    normalized_sr = {}
    for stem, path in sr_index.items():
        if sr_suffix:
            if not stem.endswith(sr_suffix):
                raise RuntimeError(
                    f'SR stem {stem!r} does not end with suffix {sr_suffix!r}.'
                )
            stem = stem[:-len(sr_suffix)]
        if stem in normalized_sr:
            raise RuntimeError(f'Duplicate normalized SR stem: {stem!r}.')
        normalized_sr[stem] = path

    missing_gt = sorted(set(normalized_sr) - set(gt_index))
    missing_sr = sorted(set(gt_index) - set(normalized_sr))
    if missing_gt or missing_sr:
        raise RuntimeError(
            'SR/GT image sets are not identical: '
            f'{len(missing_gt)} SR images lack GT and '
            f'{len(missing_sr)} GT images lack SR; '
            f'examples={missing_gt[:3] + missing_sr[:3]}.'
        )
    return [
        (stem, normalized_sr[stem], gt_index[stem])
        for stem in sorted(gt_index)
    ]


def load_rgb_tensor(path, device, crop_border=0):
    import numpy as np
    import torch
    from PIL import Image

    image = np.asarray(Image.open(path).convert('RGB')).copy()
    if crop_border:
        if min(image.shape[:2]) <= 2 * crop_border:
            raise RuntimeError(
                f'Image {path} is too small for crop_border={crop_border}.'
            )
        image = image[
            crop_border:-crop_border,
            crop_border:-crop_border,
        ]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


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


def evaluate_metric(pairs, name, metric, device, crop_border, torch):
    """Evaluate one metric while keeping only its network resident in memory."""
    protocol = METRIC_PROTOCOL[name]
    scores = []
    with torch.no_grad():
        for stem, sr_path, gt_path in pairs:
            sr = load_rgb_tensor(sr_path, device, crop_border)
            gt = load_rgb_tensor(gt_path, device, crop_border)
            if sr.shape != gt.shape:
                raise RuntimeError(
                    f'Shape mismatch for {stem}: SR={tuple(sr.shape)}, '
                    f'GT={tuple(gt.shape)}.'
                )
            score = metric(sr, gt) if protocol['reference'] else metric(sr)
            scores.append(float(score.detach().cpu().reshape(-1).mean()))
    return scores


def evaluate_metrics(pairs, metric_names, pyiqa, torch, device, crop_border):
    """Evaluate metrics sequentially to bound GPU memory use."""
    rows = [
        {
            'image': stem,
            'sr_path': str(sr_path),
            'gt_path': str(gt_path),
        }
        for stem, sr_path, gt_path in pairs
    ]
    for name in metric_names:
        metric = pyiqa.create_metric(name, device=device)
        try:
            scores = evaluate_metric(
                pairs, name, metric, device, crop_border, torch
            )
        finally:
            del metric
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        for row, score in zip(rows, scores):
            row[name] = score
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sr_root', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--method')
    parser.add_argument('--dataset')
    parser.add_argument('--checkpoint')
    parser.add_argument(
        '--metrics', nargs='+', default=['lpips', 'dists', 'musiq'],
        choices=sorted(METRIC_PROTOCOL),
    )
    parser.add_argument(
        '--sr_suffix', default='',
        help='Exact suffix appended to each GT stem in the SR output folder.',
    )
    parser.add_argument('--crop_border', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.crop_border < 0:
        raise ValueError('--crop_border must be non-negative.')
    if len(args.metrics) != len(set(args.metrics)):
        raise ValueError('--metrics must not contain duplicate names.')
    try:
        import pyiqa
        import torch
    except ImportError as error:
        raise RuntimeError(
            'PyIQA is not installed. Install the pinned journal-metric '
            'requirements in an isolated target directory.'
        ) from error

    pairs = aligned_pairs(args.sr_root, args.gt_root, args.sr_suffix)
    rows = evaluate_metrics(
        pairs, args.metrics, pyiqa, torch, args.device, args.crop_border
    )

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise RuntimeError(f'Checkpoint does not exist: {checkpoint}')
    report = {
        'schema_version': 1,
        'method': args.method,
        'dataset': args.dataset,
        'arguments': vars(args),
        'repository_commit': repository_commit(),
        'pyiqa_version': importlib.metadata.version('pyiqa'),
        'pyiqa_source': package_python_digest(pyiqa),
        'torch_version': torch.__version__,
        'metric_protocol': {
            name: METRIC_PROTOCOL[name] for name in args.metrics
        },
        'metrics': {
            name: summarize(row[name] for row in rows)
            for name in args.metrics
        },
        'artifact_sha256': {
            'sr': image_set_digest(
                (stem, sr_path) for stem, sr_path, _ in pairs
            ),
            'gt': image_set_digest(
                (stem, gt_path) for stem, _, gt_path in pairs
            ),
        },
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': file_sha256(checkpoint) if checkpoint else None,
        'per_image_csv': str(output_csv),
    }
    with output_json.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
