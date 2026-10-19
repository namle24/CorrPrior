"""Apply the frozen RealSR fidelity mechanism rules to C2/C0 and P0/C2."""

import argparse
import datetime
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.paired_metric_statistics import read_metrics
    from scripts.evaluate_preregistered_primary_endpoint import (
        stratified_resamples,
    )
except ModuleNotFoundError:  # Direct execution from DRCT/scripts.
    from paired_metric_statistics import read_metrics
    from evaluate_preregistered_primary_endpoint import stratified_resamples


METRICS = ('psnr', 'ssim', 'dists', 'lpips')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_metric(path, metric):
    table = read_metrics(path)
    missing = sorted(name for name, row in table.items() if metric not in row)
    if missing:
        raise ValueError(f'{path} lacks {metric!r} for {missing[:5]}.')
    return {name: row[metric] for name, row in table.items()}


def load_camera(specification):
    comparison, camera, base_std, cand_std, base_iqa, cand_iqa = specification
    paths = {
        'baseline_standard': str(Path(base_std).expanduser().resolve()),
        'candidate_standard': str(Path(cand_std).expanduser().resolve()),
        'baseline_iqa': str(Path(base_iqa).expanduser().resolve()),
        'candidate_iqa': str(Path(cand_iqa).expanduser().resolve()),
    }
    tables = {}
    for metric in ('psnr', 'ssim'):
        tables[f'baseline_{metric}'] = load_metric(
            paths['baseline_standard'], metric
        )
        tables[f'candidate_{metric}'] = load_metric(
            paths['candidate_standard'], metric
        )
    for metric in ('dists', 'lpips'):
        tables[f'baseline_{metric}'] = load_metric(
            paths['baseline_iqa'], metric
        )
        tables[f'candidate_{metric}'] = load_metric(
            paths['candidate_iqa'], metric
        )
    stem_sets = {key: set(table) for key, table in tables.items()}
    expected = next(iter(stem_sets.values()))
    mismatched = [key for key, stems in stem_sets.items() if stems != expected]
    if mismatched:
        raise ValueError(
            f'{comparison}/{camera} has mismatched stems in {mismatched}.'
        )
    stems = sorted(expected)
    deltas = {}
    metrics = {}
    for metric in METRICS:
        baseline = np.asarray([
            tables[f'baseline_{metric}'][stem] for stem in stems
        ], dtype=np.float64)
        candidate = np.asarray([
            tables[f'candidate_{metric}'][stem] for stem in stems
        ], dtype=np.float64)
        delta = candidate - baseline
        if not bool(np.isfinite(delta).all()):
            raise ValueError(f'{comparison}/{camera} has non-finite {metric}.')
        deltas[metric] = delta
        metrics[metric] = {
            'baseline_mean': float(baseline.mean()),
            'candidate_mean': float(candidate.mean()),
            'candidate_minus_baseline': float(delta.mean()),
        }
    return {
        'comparison': comparison,
        'camera': camera,
        'images': len(stems),
        'paths': paths,
        'sha256': {name: file_sha256(path) for name, path in paths.items()},
        'metrics': metrics,
    }, deltas


def evaluate(camera_specs, protocol):
    rule = protocol['mechanism_decision']
    expected_comparisons = set(rule['comparisons'])
    grouped = defaultdict(list)
    seen = set()
    for specification in camera_specs:
        identity = (specification[0], specification[1])
        if identity in seen:
            raise ValueError(f'Duplicate comparison/camera row: {identity}.')
        seen.add(identity)
        summary, deltas = load_camera(specification)
        grouped[specification[0]].append((summary, deltas))
    if set(grouped) != expected_comparisons:
        raise ValueError(
            f'Comparison set differs from protocol: {sorted(grouped)} vs '
            f'{sorted(expected_comparisons)}.'
        )
    camera_names = [
        {summary['camera'] for summary, _ in rows}
        for rows in grouped.values()
    ]
    if len({frozenset(names) for names in camera_names}) != 1:
        raise ValueError('Every comparison must contain the same cameras.')
    if len(camera_names[0]) < 2:
        raise ValueError('At least two camera subsets are required.')

    rng = np.random.default_rng(int(rule['seed']))
    confidence = float(rule['confidence'])
    tail = (1.0 - confidence) / 2.0
    guardrails = rule['guardrails']
    reports = {}
    for comparison in sorted(grouped):
        rows = sorted(grouped[comparison], key=lambda item: item[0]['camera'])
        psnr_arrays = [deltas['psnr'] for _, deltas in rows]
        observed = float(np.mean([values.mean() for values in psnr_arrays]))
        bootstrap = stratified_resamples(
            psnr_arrays, rng, int(rule['bootstrap_samples']), permutation=False
        )
        ci_low, ci_high = np.quantile(
            bootstrap, [tail, 1.0 - tail]
        )
        permuted = stratified_resamples(
            psnr_arrays, rng, int(rule['permutation_samples']), permutation=True
        )
        p_value = (
            float(np.count_nonzero(np.abs(permuted) >= abs(observed))) + 1.0
        ) / (int(rule['permutation_samples']) + 1.0)
        camera_guardrails = []
        for summary, deltas in rows:
            values = {
                metric: float(deltas[metric].mean())
                for metric in ('ssim', 'dists', 'lpips')
            }
            checks = {
                'ssim': (
                    values['ssim']
                    >= guardrails['ssim_candidate_minus_baseline_minimum']
                ),
                'dists': (
                    values['dists']
                    <= guardrails['dists_candidate_minus_baseline_maximum']
                ),
                'lpips': (
                    values['lpips']
                    <= guardrails['lpips_candidate_minus_baseline_maximum']
                ),
            }
            camera_guardrails.append({
                'camera': summary['camera'],
                'candidate_minus_baseline': values,
                'pass': checks,
                'all_pass': all(checks.values()),
            })
        primary_pass = (
            float(ci_low) > 0.0
            and p_value < float(rule['permutation_alpha'])
        )
        guardrails_pass = all(row['all_pass'] for row in camera_guardrails)
        reports[comparison] = {
            'claim': rule['comparisons'][comparison],
            'cameras': [summary for summary, _ in rows],
            'primary': {
                'metric': 'psnr',
                'candidate_minus_baseline': observed,
                'ci_low': float(ci_low),
                'ci_high': float(ci_high),
                'confidence': confidence,
                'permutation_p_value_two_sided': p_value,
                'pass': primary_pass,
            },
            'guardrails': camera_guardrails,
            'guardrails_pass': guardrails_pass,
            'comparison_pass': primary_pass and guardrails_pass,
        }
    return {
        'comparisons': reports,
        'claims': {
            'learned_prior_supported': reports['C2_vs_C0']['comparison_pass'],
            'spatial_gate_supported': reports['P0_vs_C2']['comparison_pass'],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument(
        '--camera', action='append', nargs=6, required=True,
        metavar=(
            'COMPARISON', 'CAMERA', 'BASE_STD', 'CAND_STD',
            'BASE_IQA', 'CAND_IQA',
        ),
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite mechanism report: {output}')
    protocol_path = args.protocol.expanduser().resolve()
    with protocol_path.open(encoding='utf-8') as file:
        protocol = json.load(file)
    decision = evaluate(args.camera, protocol)
    report = {
        'schema_version': 1,
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'disclosure': protocol['disclosure'],
        'protocol': str(protocol_path),
        'protocol_sha256': file_sha256(protocol_path),
        'implementation_sha256': file_sha256(Path(__file__).resolve()),
        **decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report['claims'], indent=2))


if __name__ == '__main__':
    main()
