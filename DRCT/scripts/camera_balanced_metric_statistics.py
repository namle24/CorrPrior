"""Camera-balanced paired uncertainty for RealSR metric tables.

Each camera subset is resampled independently and contributes equal weight to
the reported effect, even when subsets contain different image counts. This is
descriptive inference and does not apply a model-selection threshold.
"""

import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.paired_metric_statistics import metric_preference, read_metrics
except ModuleNotFoundError:  # Direct execution from DRCT/scripts.
    from paired_metric_statistics import metric_preference, read_metrics


def load_camera(specification, metrics):
    import numpy as np

    name, baseline_path, candidate_path = specification
    baseline = read_metrics(baseline_path)
    candidate = read_metrics(candidate_path)
    if set(baseline) != set(candidate):
        only_baseline = sorted(set(baseline) - set(candidate))[:5]
        only_candidate = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(
            f'Camera {name!r} has mismatched image stems. '
            f'Only baseline={only_baseline}; only candidate={only_candidate}.'
        )
    stems = sorted(baseline)
    values = {}
    for metric in metrics:
        missing = [
            stem for stem in stems
            if metric not in baseline[stem] or metric not in candidate[stem]
        ]
        if missing:
            raise ValueError(
                f'Camera {name!r} lacks metric {metric!r} for {missing[:5]}.'
            )
        baseline_values = np.asarray(
            [baseline[stem][metric] for stem in stems], dtype=np.float64
        )
        candidate_values = np.asarray(
            [candidate[stem][metric] for stem in stems], dtype=np.float64
        )
        if not (
            bool(np.isfinite(baseline_values).all())
            and bool(np.isfinite(candidate_values).all())
        ):
            raise ValueError(f'Camera {name!r} contains non-finite {metric}.')
        values[metric] = {
            'baseline': baseline_values,
            'candidate': candidate_values,
            'difference': candidate_values - baseline_values,
        }
    return {'name': name, 'images': len(stems), 'values': values}


def camera_balanced_resamples(arrays, rng, samples, permutation=False):
    import numpy as np

    result = np.zeros(samples, dtype=np.float64)
    for values in arrays:
        if permutation:
            signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
            draws = (signs * values).mean(axis=1)
        else:
            indices = rng.integers(0, len(values), size=(samples, len(values)))
            draws = values[indices].mean(axis=1)
        result += draws / len(arrays)
    return result


def evaluate(camera_specs, metrics, bootstrap_samples=10000,
             permutation_samples=10000, confidence=0.95, seed=0):
    import numpy as np

    if len(camera_specs) < 2:
        raise ValueError('At least two camera subsets are required.')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Metrics must be unique.')
    if not metrics:
        raise ValueError('At least one metric is required.')
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError('Sample counts must be positive.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('Confidence must lie strictly between zero and one.')

    cameras = [load_camera(specification, metrics) for specification in camera_specs]
    names = [camera['name'] for camera in cameras]
    if len(set(names)) != len(names):
        raise ValueError('Camera names must be unique.')

    rng = np.random.default_rng(seed)
    tail = (1.0 - confidence) / 2.0
    counts = {camera['name']: camera['images'] for camera in cameras}
    rows = []
    for metric in metrics:
        baseline_arrays = [camera['values'][metric]['baseline'] for camera in cameras]
        candidate_arrays = [
            camera['values'][metric]['candidate'] for camera in cameras
        ]
        differences = [
            camera['values'][metric]['difference'] for camera in cameras
        ]
        baseline_mean = float(np.mean([values.mean() for values in baseline_arrays]))
        candidate_mean = float(
            np.mean([values.mean() for values in candidate_arrays])
        )
        raw_difference = candidate_mean - baseline_mean
        bootstrap = camera_balanced_resamples(
            differences, rng, bootstrap_samples, permutation=False
        )
        ci_low, ci_high = np.quantile(bootstrap, [tail, 1.0 - tail])
        permuted = camera_balanced_resamples(
            differences, rng, permutation_samples, permutation=True
        )
        p_value = (
            float(np.count_nonzero(np.abs(permuted) >= abs(raw_difference))) + 1.0
        ) / (permutation_samples + 1.0)
        better = metric_preference(metric)
        direction = -1.0 if better == 'lower' else 1.0
        rows.append({
            'metric': metric,
            'better': better,
            'cameras': len(cameras),
            'images': sum(counts.values()),
            'camera_counts': json.dumps(counts, sort_keys=True),
            'weighting': 'equal_camera',
            'within_camera_resampling': 'paired_image',
            'baseline_mean': baseline_mean,
            'candidate_mean': candidate_mean,
            'candidate_minus_baseline': raw_difference,
            'favorable_effect': direction * raw_difference,
            'ci_low_raw_difference': float(ci_low),
            'ci_high_raw_difference': float(ci_high),
            'confidence': confidence,
            'permutation_p_value': p_value,
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--camera', action='append', nargs=3, required=True,
        metavar=('NAME', 'BASELINE_CSV', 'CANDIDATE_CSV'),
    )
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--permutation_samples', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = evaluate(
        args.camera,
        args.metrics,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f'Saved camera-balanced statistics to {output}')


if __name__ == '__main__':
    main()
