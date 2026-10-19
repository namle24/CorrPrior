"""Evaluate matched RealSR controls for corrected prior-learning stages."""

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_fidelity_mechanism_controls import (
        file_sha256,
        load_camera,
        stratified_resamples,
    )
except ModuleNotFoundError:  # Direct execution from DRCT/scripts.
    from evaluate_fidelity_mechanism_controls import (
        file_sha256,
        load_camera,
        stratified_resamples,
    )


def evaluate(camera_specs, protocol):
    rule = protocol['real_camera_decision']
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
    camera_sets = [
        {summary['camera'] for summary, _ in rows}
        for rows in grouped.values()
    ]
    if len({frozenset(names) for names in camera_sets}) != 1:
        raise ValueError('Every comparison must contain the same cameras.')
    if len(camera_sets[0]) < 2:
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
        ci_low, ci_high = np.quantile(bootstrap, [tail, 1.0 - tail])
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
                    values['ssim'] >=
                    guardrails['ssim_candidate_minus_baseline_minimum']
                ),
                'dists': (
                    values['dists'] <=
                    guardrails['dists_candidate_minus_baseline_maximum']
                ),
                'lpips': (
                    values['lpips'] <=
                    guardrails['lpips_candidate_minus_baseline_maximum']
                ),
            }
            camera_guardrails.append({
                'camera': summary['camera'],
                'candidate_minus_baseline': values,
                'pass': checks,
                'all_pass': all(checks.values()),
            })

        primary_pass = (
            float(ci_low) > 0.0 and
            p_value < float(rule['permutation_alpha'])
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

    claims = {
        claim: reports[comparison]['comparison_pass']
        for claim, comparison in rule['claim_keys'].items()
    }
    return {'comparisons': reports, 'claims': claims}


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
        raise RuntimeError(f'Refusing to overwrite prior-stage report: {output}')
    protocol_path = args.protocol.expanduser().resolve()
    with protocol_path.open(encoding='utf-8') as protocol_file:
        protocol = json.load(protocol_file)
    if protocol.get('schema_version') != 1:
        raise ValueError('Unexpected prior-stage protocol schema.')
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
        'dependency_sha256': file_sha256(
            Path(__file__).with_name('evaluate_fidelity_mechanism_controls.py')
        ),
        **decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write('\n')
    print(json.dumps(report['claims'], indent=2))


if __name__ == '__main__':
    main()
