"""Paired uncertainty estimates for two per-image SR metric CSV files.

The DRCT journal test configurations write one row per image.  This utility
aligns images by name, reports mean effects, obtains a paired bootstrap
confidence interval, and runs a paired random-sign permutation test.  It does
not treat images from different methods as independent samples.
"""

import argparse
import csv
import json
import os

LOWER_IS_BETTER = {'niqe', 'lpips', 'dists', 'musiq_koniq_loss'}


def metric_preference(metric):
    """Return the native optimization direction retained in result tables."""
    return 'lower' if metric.lower() in LOWER_IS_BETTER else 'higher'


def read_metrics(path):
    with open(path, newline='', encoding='utf-8') as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows or 'image' not in rows[0]:
        raise ValueError(f'{path} must contain an image column and at least one row.')
    result = {}
    for row in rows:
        image = row.pop('image')
        if image in result:
            raise ValueError(f'Duplicate image {image!r} in {path}.')
        numeric = {}
        for key, value in row.items():
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                # Provenance columns such as sr_path and gt_path are allowed,
                # but they are not candidate metrics for paired statistics.
                continue
        if not numeric:
            raise ValueError(f'No numeric metric columns found for {image!r} in {path}.')
        result[image] = numeric
    return result


def read_cluster_manifest(path, image_names):
    """Load image-to-cluster assignments from an audited dataset manifest."""
    with open(path, encoding='utf-8') as json_file:
        manifest = json.load(json_file)
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f'{path} must contain a non-empty pairs list.')
    clusters = {}
    for pair in pairs:
        if not isinstance(pair, dict) or 'stem' not in pair or 'scene' not in pair:
            raise ValueError(f'Every pair in {path} must contain stem and scene.')
        stem = str(pair['stem'])
        scene = str(pair['scene'])
        if stem in clusters:
            raise ValueError(f'Duplicate stem {stem!r} in cluster manifest {path}.')
        clusters[stem] = scene

    expected = set(image_names)
    observed = set(clusters)
    if expected != observed:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            'The cluster manifest must contain exactly the evaluated images. '
            f'Missing={missing}; extra={extra}.'
        )
    return clusters


def confidence_interval(samples, confidence):
    import numpy as np

    tail = (1.0 - confidence) / 2.0
    return np.quantile(samples, [tail, 1.0 - tail])


def paired_statistics(baseline, candidate, metric, rng, bootstrap_samples,
                      permutation_samples, confidence, clusters=None):
    import numpy as np

    image_names = sorted(baseline)
    baseline_values = np.asarray([baseline[name][metric] for name in image_names])
    candidate_values = np.asarray([candidate[name][metric] for name in image_names])
    raw_difference = candidate_values - baseline_values
    better = metric_preference(metric)
    direction = -1.0 if better == 'lower' else 1.0
    improvement = direction * raw_difference

    if clusters is None:
        cluster_labels = image_names
    else:
        cluster_labels = [clusters[name] for name in image_names]
    unique_clusters = sorted(set(cluster_labels))
    cluster_index = {label: index for index, label in enumerate(unique_clusters)}
    cluster_sums = np.zeros(len(unique_clusters), dtype=np.float64)
    cluster_counts = np.zeros(len(unique_clusters), dtype=np.int64)
    for difference, label in zip(raw_difference, cluster_labels):
        index = cluster_index[label]
        cluster_sums[index] += difference
        cluster_counts[index] += 1

    indices = rng.integers(
        0, len(unique_clusters), size=(bootstrap_samples, len(unique_clusters))
    )
    bootstrap_means = (
        cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
    )
    ci_low, ci_high = confidence_interval(bootstrap_means, confidence)

    observed = abs(improvement.mean())
    direction_cluster_sums = direction * cluster_sums
    exceedances = 0
    completed = 0
    chunk_size = 1000
    while completed < permutation_samples:
        count = min(chunk_size, permutation_samples - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(unique_clusters)))
        permuted = abs(
            (signs * direction_cluster_sums).sum(axis=1) / len(image_names)
        )
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += count
    p_value = (exceedances + 1.0) / (permutation_samples + 1.0)

    return {
        'metric': metric,
        'better': better,
        'images': len(image_names),
        'clusters': len(unique_clusters),
        'resampling_unit': 'image' if clusters is None else 'cluster',
        'baseline_mean': float(baseline_values.mean()),
        'candidate_mean': float(candidate_values.mean()),
        'candidate_minus_baseline': float(raw_difference.mean()),
        'favorable_effect': float(improvement.mean()),
        'ci_low_raw_difference': float(ci_low),
        'ci_high_raw_difference': float(ci_high),
        'confidence': confidence,
        'permutation_p_value': float(p_value),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Paired bootstrap CIs and permutation tests for SR metrics.'
    )
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--cluster_manifest',
        help='Optional JSON with pairs[].stem and pairs[].scene assignments.',
    )
    parser.add_argument('--metrics', nargs='*', default=None)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--permutation_samples', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    import numpy as np

    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError('--confidence must lie strictly between 0 and 1.')
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError('Sample counts must be positive.')

    baseline = read_metrics(args.baseline)
    candidate = read_metrics(args.candidate)
    baseline_names = set(baseline)
    candidate_names = set(candidate)
    if baseline_names != candidate_names:
        only_baseline = sorted(baseline_names - candidate_names)[:5]
        only_candidate = sorted(candidate_names - baseline_names)[:5]
        raise ValueError(
            'The CSV files must contain exactly the same images. '
            f'Only baseline={only_baseline}; only candidate={only_candidate}.'
        )
    clusters = (
        read_cluster_manifest(args.cluster_manifest, baseline_names)
        if args.cluster_manifest else None
    )

    first_name = next(iter(baseline))
    available = sorted(set(baseline[first_name]) & set(candidate[first_name]))
    metrics = args.metrics or available
    missing = sorted(set(metrics) - set(available))
    if missing:
        raise ValueError(f'Metrics absent from one or both files: {missing}.')

    rng = np.random.default_rng(args.seed)
    rows = [
        paired_statistics(
            baseline,
            candidate,
            metric,
            rng,
            args.bootstrap_samples,
            args.permutation_samples,
            args.confidence,
            clusters,
        )
        for metric in metrics
    ]
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f'Saved paired statistics to {args.output}')


if __name__ == '__main__':
    main()
