"""Apply the frozen SVSR fidelity follow-up decision rule.

The original RealSR DISTS endpoint remains unchanged. This evaluator consumes
scene-clustered paired statistics on SVSR, verifies their expected sample
counts, and applies the separately versioned follow-up rule without selecting
a model or checkpoint.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_STANDARD = {'psnr', 'ssim', 'niqe'}
EXPECTED_PYIQA = {'lpips', 'dists', 'musiq'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_statistics(path, expected_metrics, images, clusters):
    path = Path(path).expanduser().resolve()
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    observed = {row.get('metric') for row in rows}
    if observed != expected_metrics:
        raise ValueError(
            f'{path} metrics differ from the protocol: '
            f'{sorted(observed)} vs {sorted(expected_metrics)}.'
        )
    result = {}
    for row in rows:
        if int(row['images']) != images or int(row['clusters']) != clusters:
            raise ValueError(f'{path} has an unexpected image/cluster count.')
        if row['resampling_unit'] != 'cluster':
            raise ValueError(f'{path} must use scene-cluster resampling.')
        result[row['metric']] = {
            key: float(row[key])
            for key in (
                'baseline_mean',
                'candidate_mean',
                'candidate_minus_baseline',
                'ci_low_raw_difference',
                'ci_high_raw_difference',
                'confidence',
                'permutation_p_value',
            )
        }
    return path, result


def evaluate(protocol, standard, pyiqa, rule_key='unseen_confirmation'):
    confirmation = protocol[rule_key]
    guardrails = confirmation['guardrails']
    psnr = standard['psnr']
    ssim = standard['ssim']
    dists = pyiqa['dists']
    lpips = pyiqa['lpips']
    primary_ci_pass = psnr['ci_low_raw_difference'] > 0.0
    primary_permutation_pass = (
        psnr['permutation_p_value'] < confirmation['permutation_alpha']
    )
    guardrail_results = {
        'ssim': {
            'candidate_minus_baseline': ssim['candidate_minus_baseline'],
            'minimum': guardrails['ssim_candidate_minus_baseline_minimum'],
            'pass': (
                ssim['candidate_minus_baseline']
                >= guardrails['ssim_candidate_minus_baseline_minimum']
            ),
        },
        'dists': {
            'candidate_minus_baseline': dists['candidate_minus_baseline'],
            'maximum': guardrails['dists_candidate_minus_baseline_maximum'],
            'pass': (
                dists['candidate_minus_baseline']
                <= guardrails['dists_candidate_minus_baseline_maximum']
            ),
        },
        'lpips': {
            'candidate_minus_baseline': lpips['candidate_minus_baseline'],
            'maximum': guardrails['lpips_candidate_minus_baseline_maximum'],
            'pass': (
                lpips['candidate_minus_baseline']
                <= guardrails['lpips_candidate_minus_baseline_maximum']
            ),
        },
    }
    guardrails_pass = all(item['pass'] for item in guardrail_results.values())
    return {
        'primary': {
            'metric': 'psnr',
            **psnr,
            'ci_pass': primary_ci_pass,
            'permutation_pass': primary_permutation_pass,
            'pass': primary_ci_pass and primary_permutation_pass,
        },
        'guardrails': guardrail_results,
        'guardrails_pass': guardrails_pass,
        'confirmation_pass': (
            primary_ci_pass and primary_permutation_pass and guardrails_pass
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--standard_statistics', type=Path, required=True)
    parser.add_argument('--pyiqa_statistics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis_name', default='P0 versus B0 on SVSR')
    parser.add_argument('--rule_key', default='unseen_confirmation')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite endpoint report: {args.output}')
    protocol_path = args.protocol.expanduser().resolve()
    with protocol_path.open(encoding='utf-8') as file:
        protocol = json.load(file)
    if protocol.get('schema_version') != 1:
        raise ValueError('Unexpected fidelity follow-up protocol schema.')
    if args.rule_key not in protocol:
        raise ValueError(f'Unknown fidelity decision rule: {args.rule_key}')
    confirmation = protocol[args.rule_key]
    images = int(confirmation['images'])
    clusters = int(confirmation['scenes'])
    standard_path, standard = read_statistics(
        args.standard_statistics, EXPECTED_STANDARD, images, clusters
    )
    pyiqa_path, pyiqa = read_statistics(
        args.pyiqa_statistics, EXPECTED_PYIQA, images, clusters
    )
    decision = evaluate(protocol, standard, pyiqa, rule_key=args.rule_key)
    report = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_name': args.analysis_name,
        'decision_rule_key': args.rule_key,
        'disclosure': protocol['disclosure'],
        'protocol': str(protocol_path),
        'protocol_sha256': file_sha256(protocol_path),
        'implementation_sha256': file_sha256(Path(__file__).resolve()),
        'input_sha256': {
            'standard_statistics': file_sha256(standard_path),
            'pyiqa_statistics': file_sha256(pyiqa_path),
        },
        'standard_statistics': standard,
        'pyiqa_statistics': pyiqa,
        **decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({
        'primary': report['primary'],
        'guardrails_pass': report['guardrails_pass'],
        'confirmation_pass': report['confirmation_pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
