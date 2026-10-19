"""Evaluate the preregistered camera-balanced P0-versus-B0 endpoint.

Each camera contributes equally regardless of image count. DISTS is tested
with a paired stratified bootstrap, while PSNR and SSIM remain per-camera
distortion guardrails. The decision rule is fixed in the journal protocol and
must not be changed after inspecting downstream results.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.paired_metric_statistics import read_metrics
except ModuleNotFoundError:  # Direct execution from DRCT/scripts.
    from paired_metric_statistics import read_metrics


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def required_metric(path, metric):
    table = read_metrics(path)
    missing = sorted(name for name, row in table.items() if metric not in row)
    if missing:
        raise ValueError(
            f'Metric {metric!r} is absent for images {missing[:5]} in {path}.'
        )
    return {name: row[metric] for name, row in table.items()}


def load_camera(specification):
    name, baseline_standard, candidate_standard, baseline_iqa, candidate_iqa = (
        specification
    )
    paths = {
        'baseline_standard': str(Path(baseline_standard).expanduser().resolve()),
        'candidate_standard': str(Path(candidate_standard).expanduser().resolve()),
        'baseline_iqa': str(Path(baseline_iqa).expanduser().resolve()),
        'candidate_iqa': str(Path(candidate_iqa).expanduser().resolve()),
    }
    tables = {
        'baseline_psnr': required_metric(paths['baseline_standard'], 'psnr'),
        'candidate_psnr': required_metric(paths['candidate_standard'], 'psnr'),
        'baseline_ssim': required_metric(paths['baseline_standard'], 'ssim'),
        'candidate_ssim': required_metric(paths['candidate_standard'], 'ssim'),
        'baseline_dists': required_metric(paths['baseline_iqa'], 'dists'),
        'candidate_dists': required_metric(paths['candidate_iqa'], 'dists'),
    }
    stem_sets = {key: set(table) for key, table in tables.items()}
    expected = next(iter(stem_sets.values()))
    mismatched = [key for key, stems in stem_sets.items() if stems != expected]
    if mismatched:
        raise ValueError(
            f'Camera {name!r} has mismatched image stems in {mismatched}.'
        )

    stems = sorted(expected)
    deltas = {
        metric: np.asarray([
            tables[f'candidate_{metric}'][stem]
            - tables[f'baseline_{metric}'][stem]
            for stem in stems
        ], dtype=np.float64)
        for metric in ('dists', 'psnr', 'ssim')
    }
    for metric, values in deltas.items():
        if not bool(np.isfinite(values).all()):
            raise ValueError(f'Camera {name!r} contains non-finite {metric}.')

    summary = {
        'name': name,
        'images': len(stems),
        'paths': paths,
        'sha256': {key: file_sha256(path) for key, path in paths.items()},
        'metrics': {},
    }
    for metric in ('dists', 'psnr', 'ssim'):
        baseline = np.asarray([
            tables[f'baseline_{metric}'][stem] for stem in stems
        ], dtype=np.float64)
        candidate = np.asarray([
            tables[f'candidate_{metric}'][stem] for stem in stems
        ], dtype=np.float64)
        summary['metrics'][metric] = {
            'baseline_mean': float(baseline.mean()),
            'candidate_mean': float(candidate.mean()),
            'candidate_minus_baseline': float(deltas[metric].mean()),
        }
    return summary, deltas


def stratified_resamples(arrays, rng, samples, permutation=False):
    """Return equal-camera-weighted paired bootstrap or sign-flip draws."""
    result = np.zeros(samples, dtype=np.float64)
    camera_weight = 1.0 / len(arrays)
    for values in arrays:
        if permutation:
            signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
            draws = (signs * values).mean(axis=1)
        else:
            indices = rng.integers(
                0, len(values), size=(samples, len(values))
            )
            draws = values[indices].mean(axis=1)
        result += camera_weight * draws
    return result


def evaluate(camera_specs, bootstrap_samples=10000, permutation_samples=10000,
             confidence=0.95, seed=0, psnr_tolerance=0.05,
             ssim_tolerance=0.001):
    if len(camera_specs) < 2:
        raise ValueError('At least two camera subsets are required.')
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError('Resampling counts must be positive.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('Confidence must lie strictly between zero and one.')
    if psnr_tolerance < 0.0 or ssim_tolerance < 0.0:
        raise ValueError('Guardrail tolerances must be non-negative.')

    cameras = []
    camera_deltas = []
    seen_names = set()
    for specification in camera_specs:
        if specification[0] in seen_names:
            raise ValueError(f'Duplicate camera name {specification[0]!r}.')
        seen_names.add(specification[0])
        summary, deltas = load_camera(specification)
        cameras.append(summary)
        camera_deltas.append(deltas)

    dists_arrays = [deltas['dists'] for deltas in camera_deltas]
    observed = float(np.mean([values.mean() for values in dists_arrays]))
    rng = np.random.default_rng(seed)
    bootstrap = stratified_resamples(
        dists_arrays, rng, bootstrap_samples, permutation=False
    )
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(bootstrap, [tail, 1.0 - tail])
    permuted = stratified_resamples(
        dists_arrays, rng, permutation_samples, permutation=True
    )
    p_value = (
        float(np.count_nonzero(np.abs(permuted) >= abs(observed))) + 1.0
    ) / (permutation_samples + 1.0)

    guardrails = []
    for camera, deltas in zip(cameras, camera_deltas):
        psnr_delta = float(deltas['psnr'].mean())
        ssim_delta = float(deltas['ssim'].mean())
        guardrails.append({
            'camera': camera['name'],
            'psnr_candidate_minus_baseline': psnr_delta,
            'psnr_minimum_allowed': -psnr_tolerance,
            'psnr_pass': psnr_delta >= -psnr_tolerance,
            'ssim_candidate_minus_baseline': ssim_delta,
            'ssim_minimum_allowed': -ssim_tolerance,
            'ssim_pass': ssim_delta >= -ssim_tolerance,
        })

    primary_pass = float(ci_high) < 0.0
    guardrails_pass = all(
        item['psnr_pass'] and item['ssim_pass'] for item in guardrails
    )
    return {
        'schema_version': 1,
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'implementation_sha256': file_sha256(Path(__file__).resolve()),
        'decision_rule': {
            'primary_metric': 'camera-balanced mean per-image DISTS',
            'effect': 'candidate_minus_baseline',
            'success_condition': 'upper paired-bootstrap CI endpoint < 0',
            'camera_weighting': 'equal',
            'within_camera_resampling': 'paired image',
            'confidence': confidence,
            'bootstrap_samples': bootstrap_samples,
            'permutation_samples': permutation_samples,
            'seed': seed,
            'psnr_loss_tolerance_db_per_camera': psnr_tolerance,
            'ssim_loss_tolerance_per_camera': ssim_tolerance,
        },
        'cameras': cameras,
        'primary': {
            'candidate_minus_baseline': observed,
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'permutation_p_value_two_sided': p_value,
            'pass': primary_pass,
        },
        'guardrails': guardrails,
        'guardrails_pass': guardrails_pass,
        'screen_pass': primary_pass and guardrails_pass,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--camera', action='append', nargs=5, required=True,
        metavar=('NAME', 'BASE_STD', 'CAND_STD', 'BASE_IQA', 'CAND_IQA'),
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--permutation_samples', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--psnr_tolerance', type=float, default=0.05)
    parser.add_argument('--ssim_tolerance', type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    report = evaluate(
        args.camera,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        confidence=args.confidence,
        seed=args.seed,
        psnr_tolerance=args.psnr_tolerance,
        ssim_tolerance=args.ssim_tolerance,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite primary report: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({
        'primary': report['primary'],
        'guardrails_pass': report['guardrails_pass'],
        'screen_pass': report['screen_pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
